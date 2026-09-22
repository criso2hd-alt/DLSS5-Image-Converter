"""The 3D tab's effects panel: add effects, pick one, edit and keyframe it.

Every control is a slider row with a keyframe diamond, in the same style as
the rest of the app. The diamond works the way it does in After Effects:

- hollow and dim: the property is not animated; editing changes it everywhere.
- hollow and lit: the property is animated, but not keyed at the playhead.
  Editing it adds a key here (auto-key), so animating needs no extra clicks.
- filled: keyed at the playhead. Clicking removes that key.

The panel edits the base EffectsState for anything unanimated and the
EffectsTrack for anything animated; it never owns either. The page owns
them and is told about changes through `owner.fx_changed()`.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QMenu, QPushButton, QSlider,
                               QToolButton, QVBoxLayout, QWidget)

from .effects3d import (LightningStrike, emitter_preset, plane_preset, volume_preset)
from .widgets import ModuleCard

VOLUME_TYPES = ["Fog", "Smoke", "Fire", "Cloud", "Godrays"]
PARTICLE_TYPES = ["Embers", "Dust", "Snow", "Rain", "Smoke", "Fire", "Clouds"]
PLANE_TYPES = ["Floor", "Ceiling", "Wall"]

#: Named particle directions as (tilt, heading) in degrees. Tilt 0 is scene up;
#: heading 0 is away from the camera, 90 to its right.
DIRECTIONS = {
    "Up": (0.0, 0.0),
    "Down": (180.0, 0.0),
    "Left": (90.0, 270.0),
    "Right": (90.0, 90.0),
    "Toward camera": (90.0, 180.0),
    "Away from camera": (90.0, 0.0),
}


def angles_to_direction(tilt: float, heading: float) -> tuple[float, float, float]:
    t, hd = math.radians(tilt), math.radians(heading)
    return (math.sin(t) * math.sin(hd), math.cos(t), -math.sin(t) * math.cos(hd))


def direction_to_angles(direction) -> tuple[float, float]:
    x, y, z = (float(v) for v in direction)
    length = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / length, y / length, z / length
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, y))))
    heading = math.degrees(math.atan2(x, -z)) % 360.0 if abs(y) < 0.9999 else 0.0
    return tilt, heading


def direction_name(direction) -> str:
    tilt, heading = direction_to_angles(direction)
    for name, (t, hd) in DIRECTIONS.items():
        dh = abs((heading - hd + 180.0) % 360.0 - 180.0)
        if abs(tilt - t) < 0.5 and (dh < 0.5 or t in (0.0, 180.0)):
            return name
    return "Custom"


def kind_of(item) -> str:
    """volume, particles, lightning or plane."""
    if isinstance(item, LightningStrike):
        return "lightning"
    if getattr(item, "kind", "") in ("floor", "ceiling", "wall"):
        return "plane"
    return "volume" if hasattr(item, "density") else "particles"


#: Height of one row in the effects list.
ROW_H = 32

GLYPHS = {"volume": "☁", "particles": "✦", "lightning": "⚡", "plane": "▭"}
TYPE_TAGS = {"volume": "VOLUME", "particles": "PARTICLES", "lightning": "WEATHER", "plane": "SURFACE"}


class Prop:
    """One slider: which field, its range and how it reads."""

    def __init__(self, field: str, label: str, lo: float, hi: float, decimals: int = 2,
                 suffix: str = "", tip: str = "", curve: float = 1.0) -> None:
        self.field, self.label, self.lo, self.hi = field, label, lo, hi
        self.decimals, self.suffix, self.tip, self.curve = decimals, suffix, tip, curve


#: Sections of sliders per effect type. Colour, direction and on/off switches
#: are added separately because they are not single numbers.
SPECS: dict[str, list[tuple[str, list[Prop]]]] = {
    "volume": [
        ("Look", [
            Prop("density", "Density", 0.0, 3.0, tip="How thick the volume is."),
            Prop("emission", "Glow", 0.0, 10.0, 1, tip="Light the volume gives off itself."),
            Prop("light_response", "Light response", 0.0, 2.0,
                 tip="How strongly the scene lighting shades it."),
        ]),
        ("Motion", [
            Prop("speed", "Drift", -3.0, 3.0, tip="How fast the noise flows."),
            Prop("noise_scale", "Noise scale", 0.1, 8.0, tip="Size of the swirls."),
            Prop("detail", "Detail", 0.0, 1.0, tip="Fine breakup on top of the swirls."),
        ]),
    ],
    "particles": [
        ("Emission", [
            Prop("count", "Count", 0, 20000, 0, curve=2.0, tip="How many particles are alive."),
            Prop("lifetime", "Lifetime", 0.1, 20.0, 1, " s", "How long each particle lives."),
            Prop("rate", "Time scale", 0.0, 4.0, tip="Plays the whole emitter faster or slower."),
            Prop("particle_size", "Particle size", 0.002, 1.0, 3, curve=2.0),
            Prop("opacity", "Opacity", 0.0, 1.0),
        ]),
        ("Motion", [
            Prop("speed", "Launch speed", -10.0, 10.0, tip="Speed along the direction at birth."),
            Prop("gravity", "Lift", -10.0, 10.0,
                 tip="Keeps pushing along the direction (buoyancy); negative pulls back."),
            Prop("spread", "Spread", 0.0, 3.0, tip="How wide the stream fans out."),
            Prop("drag", "Drag", 0.0, 5.0, tip="Air resistance: high values billow and stop."),
            Prop("turbulence", "Turbulence", 0.0, 4.0, tip="Swirl and curl."),
            Prop("growth", "Growth", 0.0, 5.0, tip="How much each particle swells as it ages."),
            Prop("bounce", "Bounce", 0.0, 1.0,
                 tip="0 slides along floors and walls, 1 bounces straight back."),
        ]),
        ("Look", [
            Prop("emission", "Glow", 0.0, 10.0, 1),
            Prop("light_response", "Light response", 0.0, 2.0),
        ]),
    ],
    "lightning": [
        ("Strikes", [
            Prop("rate", "Random strikes per minute", 0.0, 120.0, 1, curve=2.0,
                 tip="On average, at irregular moments like a real storm. 0 turns random "
                     "strikes off, leaving only the ones you place."),
            Prop("height", "Height", 0.5, 30.0, 1, tip="How far above the target the bolt starts."),
            Prop("seed", "Variation", 0.0, 100.0, 0,
                 tip="A different storm: other strike times and bolt shapes."),
        ]),
        ("Look", [
            Prop("emission", "Bolt brightness", 0.0, 20.0, 1),
            Prop("flash", "Scene flash", 0.0, 3.0,
                 tip="How much the whole frame lights up during a strike."),
        ]),
    ],
    "plane": [],
}


#: The scene-wide wet-surface controls (fields of LightingSettings).
WET_PROPS = [
    Prop("wetness", "Wetness", 0.0, 1.0,
         tip="Darkens surfaces and makes the ground shine, as after rain."),
    Prop("puddles", "Puddles", 0.0, 1.0, tip="How much of the ground is standing water."),
    Prop("puddle_size", "Puddle size", 0.2, 5.0, curve=2.0, tip="Size of the puddle patches."),
    Prop("ripples", "Rain ripples", 0.0, 1.0,
         tip="Rings from raindrops disturbing the reflections."),
]


class KeyDiamond(QToolButton):
    """Hollow dim diamond when unanimated, hollow lit when animated elsewhere,
    filled when keyed at the playhead. Painted rather than drawn from a font
    glyph: ◇ renders at wildly different sizes depending on the font."""

    STYLES = {
        "none": ("#56607a", False, "Not animated. Click to add a keyframe at the playhead."),
        "animated": ("#8b79ff", False, "Animated. Click, or change the value, to key it here."),
        "keyed": ("#b7adff", True, "Keyed at the playhead. Click to remove this keyframe."),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(22, 22)
        self._hover = False
        self.set_state("none")

    def set_state(self, state: str) -> None:
        self._colour, self._filled, tip = self.STYLES[state]
        self.setToolTip(tip)
        self.update()

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPainter, QPen, QPolygonF
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = QColor("#ffffff" if self._hover else self._colour)
        cx, cy, r = self.width() / 2, self.height() / 2, 5.5
        shape = QPolygonF([QPointF(cx, cy - r), QPointF(cx + r, cy),
                           QPointF(cx, cy + r), QPointF(cx - r, cy)])
        p.setPen(QPen(colour, 1.6))
        p.setBrush(colour if self._filled else Qt.BrushStyle.NoBrush)
        p.drawPolygon(shape)


class EyeButton(QToolButton):
    """Open eye: the gizmo shows in the 3D view. Crossed out: hidden."""

    toggled_eye = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAutoRaise(True)
        self.setCheckable(True)
        self.setChecked(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(24, 22)
        self._hover = False
        self.toggled.connect(self._changed)
        self._changed(True)

    def _changed(self, on: bool) -> None:
        self.setToolTip("Gizmo shown in the 3D view. Click to hide it (it still renders)."
                        if on else "Gizmo hidden. Click to show it in the 3D view again.")
        self.update()
        self.toggled_eye.emit(on)

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import QPainter, QPainterPath, QPen
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        on = self.isChecked()
        colour = QColor("#ffffff" if self._hover else ("#9fb0cc" if on else "#4d566b"))
        cx, cy = self.width() / 2, self.height() / 2
        eye = QPainterPath()
        eye.moveTo(cx - 8, cy)
        eye.quadTo(cx, cy - 7, cx + 8, cy)
        eye.quadTo(cx, cy + 7, cx - 8, cy)
        p.setPen(QPen(colour, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(eye)
        p.setBrush(colour)
        p.drawEllipse(QRectF(cx - 2.4, cy - 2.4, 4.8, 4.8))
        if not on:
            p.setPen(QPen(colour, 1.8))
            p.drawLine(QPointF(cx - 7, cy + 6), QPointF(cx + 7, cy - 6))


class _ListRow(QWidget):
    """One effect in the list: on/off tick, icon and name, gizmo eye."""

    def __init__(self, text: str, enabled: bool, shown: bool, parent: QWidget | None = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 0, 4, 0)
        row.setSpacing(6)
        self.tick = QCheckBox()
        self.tick.setChecked(enabled)
        self.tick.setToolTip("Render this effect. Untick to switch it off without removing it.")
        label = QLabel(text)
        # Clicks on the name must reach the list, which does the selecting.
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.eye = EyeButton()
        self.eye.setChecked(shown)
        row.addWidget(self.tick)
        row.addWidget(label, 1)
        row.addWidget(self.eye)


class FxSlider(QWidget):
    """Name, editable value and keyframe diamond over a full-width slider."""

    changed = Signal(float)
    key_clicked = Signal()

    STEPS = 1000

    def __init__(self, prop: Prop, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.prop = prop
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 2, 0, 2)
        col.setSpacing(2)
        head = QHBoxLayout()
        head.setSpacing(4)
        name = QLabel(prop.label)
        self.value = QDoubleSpinBox()
        self.value.setRange(prop.lo, prop.hi)
        self.value.setDecimals(prop.decimals)
        self.value.setSuffix(prop.suffix)
        self.value.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value.setFixedWidth(74)
        self.value.setKeyboardTracking(False)
        self.value.valueChanged.connect(self._typed)
        self.diamond = KeyDiamond()
        self.diamond.clicked.connect(self.key_clicked.emit)
        head.addWidget(name)
        head.addStretch(1)
        head.addWidget(self.value)
        head.addWidget(self.diamond)
        col.addLayout(head)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.slider.valueChanged.connect(self._slid)
        col.addWidget(self.slider)
        if prop.tip:
            self.setToolTip(prop.tip)

    def _to_raw(self, v: float) -> int:
        p = self.prop
        t = (float(v) - p.lo) / max(p.hi - p.lo, 1e-9)
        t = max(0.0, min(1.0, t)) ** (1.0 / p.curve)
        return int(round(t * self.STEPS))

    def _from_raw(self, raw: int) -> float:
        p = self.prop
        return p.lo + (raw / self.STEPS) ** p.curve * (p.hi - p.lo)

    def _slid(self, raw: int) -> None:
        v = round(self._from_raw(raw), self.prop.decimals)
        self.value.blockSignals(True)
        self.value.setValue(v)
        self.value.blockSignals(False)
        self.changed.emit(v)

    def _typed(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_raw(v))
        self.slider.blockSignals(False)
        self.changed.emit(v)

    def set_value(self, v: float) -> None:
        for w in (self.value, self.slider):
            w.blockSignals(True)
        self.value.setValue(float(v))
        self.slider.setValue(self._to_raw(v))
        for w in (self.value, self.slider):
            w.blockSignals(False)


class _Row(QWidget):
    """A label, a control and a keyframe diamond on one line."""

    def __init__(self, label: str, control: QWidget | None, keyed: bool = True,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(4)
        row.addWidget(QLabel(label))
        row.addStretch(1)
        if control is not None:
            row.addWidget(control)
        self.diamond = KeyDiamond() if keyed else None
        if self.diamond is not None:
            row.addWidget(self.diamond)


class FxPanel(QWidget):
    """Effects list plus the selected effect's properties, as two cards."""

    def __init__(self, owner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.owner = owner
        self._item_id: str | None = None
        #: Effects whose gizmo is switched off in the 3D view (editor only).
        self._hidden: set[str] = set()
        self._sliders: dict[str, FxSlider] = {}
        self._rows: dict[str, _Row] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        # -- the list -----------------------------------------------------
        self.list_card = ModuleCard("Effects")
        buttons = QHBoxLayout()
        self.add_button = QPushButton("+  Add effect")
        self.add_button.setMenu(self._add_menu())
        self.add_button.setToolTip("Volumes, particles, lightning, and floor or wall planes "
                                   "that particles collide with.")
        self.key_all = QPushButton("◆  Key all")
        self.key_all.setObjectName("secondary")
        self.key_all.setToolTip("Keyframe every effect's current settings at the playhead.")
        self.key_all.clicked.connect(self._key_all)
        self.eye_all = EyeButton()
        self.eye_all.setToolTip("Show or hide every effect's gizmo in the 3D view.")
        self.eye_all.clicked.connect(self._eye_all_clicked)
        buttons.addWidget(self.add_button, 1)
        buttons.addWidget(self.key_all)
        buttons.addWidget(self.eye_all)
        self.list_card.add_layout(buttons)
        self.list = QListWidget()
        # Selection as a quiet panel with an accent edge, like the tabs; the
        # global bright highlight made white text unreadable here.
        self.list.setStyleSheet(
            "QListWidget { background: transparent; border: none; outline: none; }"
            "QListWidget::item { margin: 1px 0; border-radius: 6px; color: #c9d1e0; }"
            "QListWidget::item:hover { background: #172033; }"
            "QListWidget::item:selected { background: #1d2940; color: #ffffff;"
            " border-left: 3px solid #35d6f5; }")
        self.list.currentItemChanged.connect(lambda *_: self._selected())
        self.list_card.add(self.list)
        self.empty_hint = QLabel("Add fog, particles, rain or lightning, then place them "
                                 "in the 3D view with the gizmo.")
        self.empty_hint.setObjectName("hint")
        self.empty_hint.setWordWrap(True)
        self.list_card.add(self.empty_hint)
        outer.addWidget(self.list_card)

        # -- the selected effect -----------------------------------------
        self.props_card = ModuleCard("Properties")
        self.props_host = QWidget()
        self.props_layout = QVBoxLayout(self.props_host)
        self.props_layout.setContentsMargins(0, 0, 0, 0)
        self.props_layout.setSpacing(6)
        self.props_card.add(self.props_host)
        self.props_card.setVisible(False)
        outer.addWidget(self.props_card)

        # -- wet surfaces: scene-wide, so a card of its own ------------------
        self.wet_card = ModuleCard("Wet surfaces")
        self._env_sliders: dict[str, FxSlider] = {}
        for prop in WET_PROPS:
            s = FxSlider(prop)
            s.changed.connect(lambda v, f=prop.field: self.set_env(f, v))
            s.key_clicked.connect(lambda f=prop.field: self._toggle_env_key(f))
            self._env_sliders[prop.field] = s
            self.wet_card.add(s)
        note = QLabel("Wet ground reflects the scene, most strongly at low angles. "
                      "Reflections only show what is in frame.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        self.wet_card.add(note)
        outer.addWidget(self.wet_card)
        self._refresh_list()

    # -- adding ------------------------------------------------------------

    def _add_menu(self) -> QMenu:
        menu = QMenu(self)
        groups = (("Volumes", VOLUME_TYPES, self._add_volume),
                  ("Particles", PARTICLE_TYPES, self._add_emitter),
                  ("Weather", ["Lightning"], lambda _k: self._add_lightning()),
                  ("Collision surfaces", PLANE_TYPES, self._add_plane))
        for title, names, action in groups:
            head = menu.addAction(title.upper())
            head.setEnabled(False)
            for name in names:
                menu.addAction(f"   {name}", lambda n=name, a=action: a(n))
            menu.addSeparator()
        return menu

    def _add_volume(self, name: str) -> None:
        item = volume_preset(name, self.owner.pivot_z())
        self.owner.effects.volumes.append(item)
        self._added(item)

    def _add_emitter(self, name: str) -> None:
        item = emitter_preset(name, self.owner.pivot_z())
        self.owner.effects.emitters.append(item)
        self._added(item)

    def _add_lightning(self) -> None:
        z = self.owner.pivot_z()
        ground = self.owner.ground_plane()
        y = -0.5
        if ground is not None:
            n, c = ground
            if abs(float(n[1])) > 1e-3:
                y = -(float(c) + float(n[2]) * z) / float(n[1])
        item = LightningStrike(position=(0.0, y, z))
        self.owner.effects.strikes.append(item)
        self._added(item)

    def _add_plane(self, name: str) -> None:
        item = plane_preset(name, self.owner.pivot_z(), self.owner.ground_plane())
        self.owner.effects.planes.append(item)
        self._added(item)

    def _added(self, item) -> None:
        self._item_id = item.id
        self._refresh_list()
        self.owner.fx_changed()

    # -- list --------------------------------------------------------------

    def _refresh_list(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        current = None
        for item in self.owner.effects.items():
            li = QListWidgetItem()
            li.setData(Qt.ItemDataRole.UserRole, item.id)
            self.list.addItem(li)
            row = _ListRow(f"{GLYPHS[kind_of(item)]}   {item.name}", item.enabled,
                           item.id not in self._hidden)
            row.tick.toggled.connect(lambda on, i=item.id: self._set_enabled(i, on))
            row.eye.toggled_eye.connect(lambda on, i=item.id: self._set_shown(i, on))
            li.setSizeHint(QSize(0, ROW_H))
            self.list.setItemWidget(li, row)
            if item.id == self._item_id:
                current = li
        self.list.blockSignals(False)
        has = self.list.count() > 0
        self.list.setVisible(has)
        # Exactly as tall as its rows (up to six), so the card never shows an
        # empty well under a short list.
        self.list.setFixedHeight(min(self.list.count(), 6) * (ROW_H + 2) + 6)
        self.empty_hint.setVisible(not has)
        if current is not None:
            self.list.setCurrentItem(current)
        else:
            self._item_id = None
        self._selected()

    def _set_enabled(self, item_id: str, on: bool) -> None:
        item = self.owner.effects.item(item_id)
        if item is not None:
            item.enabled = on
            self.owner.fx_changed()

    def _set_shown(self, item_id: str, on: bool) -> None:
        if on:
            self._hidden.discard(item_id)
        else:
            self._hidden.add(item_id)
        self.owner.viewport.set_hidden_effects(self._hidden)
        # The header eye is open until every gizmo is hidden, so its first
        # click always declutters (hides the rest) rather than undoing work.
        everything = {i.id for i in self.owner.effects.items()}
        self.eye_all.blockSignals(True)
        self.eye_all.setChecked(not everything or not everything <= self._hidden)
        self.eye_all.blockSignals(False)

    def _eye_all_clicked(self) -> None:
        on = self.eye_all.isChecked()
        self._hidden = set() if on else {i.id for i in self.owner.effects.items()}
        self.owner.viewport.set_hidden_effects(self._hidden)
        self._refresh_list()

    def select(self, item_id: str) -> None:
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == item_id:
                self.list.setCurrentRow(i)

    def _selected(self) -> None:
        li = self.list.currentItem()
        self._item_id = li.data(Qt.ItemDataRole.UserRole) if li is not None else None
        self.owner.viewport.set_effect_selection(self._item_id)
        self._build_props()

    def base_item(self):
        return self.owner.effects.item(self._item_id) if self._item_id else None

    # -- properties ----------------------------------------------------------

    def _clear_props(self) -> None:
        while self.props_layout.count():
            child = self.props_layout.takeAt(0)
            w = child.widget()
            if w is not None:
                # Detach now: deleteLater alone leaves the old rows painted
                # over the new ones until the event loop gets round to them.
                w.hide()
                w.setParent(None)
                w.deleteLater()
            elif child.layout() is not None:
                QWidget().setLayout(child.layout())   # re-parent to delete
        self._sliders.clear()
        self._rows.clear()

    def _section(self, title: str) -> None:
        lab = QLabel(title.upper())
        lab.setObjectName("modTag")
        lab.setStyleSheet("color: #7f8aa3; letter-spacing: 1.5px; font-size: 10px;"
                          "padding-top: 6px;")
        self.props_layout.addWidget(lab)

    def _build_props(self) -> None:
        self._clear_props()
        item = self.base_item()
        self.props_card.setVisible(item is not None)
        if item is None:
            return
        kind = kind_of(item)
        self.props_card.title_label.setText(item.name.upper())

        # Place: moved with the gizmo; one diamond keys position and turn.
        self._section("Placement")
        place = _Row("Position & rotation", None)
        place.setToolTip("Drag the effect in the 3D view to move it. The diamond keys "
                         "where it is now, so two keys make it travel.")
        place.diamond.clicked.connect(lambda: self._toggle_key(("position", "rotation")))
        self._rows["position"] = place
        self.props_layout.addWidget(place)
        size = FxSlider(Prop("size", "Size", 0.05, 20.0, 2, curve=2.0,
                             tip="Scales the effect's box (for a plane, only how big it is "
                                 "drawn: it collides everywhere)."))
        size.changed.connect(self._set_size)
        size.key_clicked.connect(lambda: self._toggle_key(("size",)))
        self._sliders["size"] = size
        self.props_layout.addWidget(size)

        for title, props in SPECS[kind]:
            self._section(title)
            if title == "Motion" and kind == "particles":
                self._direction_rows()
            for prop in props:
                slider = FxSlider(prop)
                slider.changed.connect(lambda v, f=prop.field: self.set_value(f, v))
                slider.key_clicked.connect(lambda f=prop.field: self._toggle_key((f,)))
                self._sliders[prop.field] = slider
                self.props_layout.addWidget(slider)
            if title == "Look" and kind != "plane":
                self._colour_row()

        if kind == "particles":
            collide = QCheckBox("Collide with floor, walls and ceiling")
            collide.setChecked(bool(getattr(item, "collide", True)))
            collide.toggled.connect(lambda v: self.set_value("collide", v))
            self.props_layout.addWidget(collide)
        if kind == "lightning":
            self._strike_rows(item)
            bolt = QCheckBox("Show the bolt")
            bolt.setToolTip("Off: only the flash, like lightning striking out of shot.")
            bolt.setChecked(bool(item.show_bolt))
            bolt.toggled.connect(lambda v: self.set_value("show_bolt", v))
            self.props_layout.addWidget(bolt)

        actions = QHBoxLayout()
        dup = QPushButton("Duplicate")
        dup.setObjectName("secondary")
        dup.clicked.connect(self._duplicate)
        rem = QPushButton("Remove")
        rem.setObjectName("secondary")
        rem.clicked.connect(self._remove)
        actions.addWidget(dup)
        actions.addWidget(rem)
        holder = QWidget()
        holder.setLayout(actions)
        self.props_layout.addWidget(holder)
        self.sync()

    def _direction_rows(self) -> None:
        combo = QComboBox()
        combo.addItems(list(DIRECTIONS) + ["Custom"])
        combo.setToolTip("Which way the particles travel. The photo decides what 'up' is, "
                         "so pick it here, or use Custom and set the angles.")
        combo.currentTextChanged.connect(self._direction_preset)
        row = _Row("Direction", combo)
        row.diamond.clicked.connect(lambda: self._toggle_key(("direction",)))
        row.combo = combo
        self._rows["direction"] = row
        self.props_layout.addWidget(row)
        for field, label, hi, tip in (("tilt", "Tilt", 180.0, "0° travels up, 90° sideways, 180° down."),
                                      ("heading", "Heading", 360.0,
                                       "0° away from the camera, 90° right, 180° toward it.")):
            s = FxSlider(Prop(field, label, 0.0, hi, 0, "°", tip))
            s.diamond.hide()        # keyed together through the Direction diamond
            s.changed.connect(lambda _v: self._direction_angles())
            self._sliders[field] = s
            self.props_layout.addWidget(s)

    def _strike_rows(self, item) -> None:
        """Placed strikes: exact moments, shown as ticks on the timeline."""
        self._section("Placed strikes")
        buttons = QHBoxLayout()
        add = QPushButton("⚡  Strike at playhead")
        add.setToolTip("Adds a strike at exactly this moment, on top of any random ones. "
                       "Set random strikes to 0 to time every strike yourself.")
        add.clicked.connect(self._add_strike)
        remove = QPushButton("Remove")
        remove.setObjectName("secondary")
        remove.setToolTip("Removes the placed strike at (or just before) the playhead.")
        remove.clicked.connect(self._remove_strike)
        clear = QPushButton("Clear")
        clear.setObjectName("secondary")
        clear.setToolTip("Removes every placed strike.")
        clear.clicked.connect(self._clear_strikes)
        buttons.addWidget(add, 1)
        buttons.addWidget(remove)
        buttons.addWidget(clear)
        holder = QWidget()
        holder.setLayout(buttons)
        self.props_layout.addWidget(holder)
        self.strike_label = QLabel("")
        self.strike_label.setObjectName("hint")
        self.strike_label.setWordWrap(True)
        self.props_layout.addWidget(self.strike_label)
        self._show_strikes()

    def _show_strikes(self) -> None:
        item = self.base_item()
        if item is None or kind_of(item) != "lightning" or not hasattr(self, "strike_label"):
            return
        times = sorted(item.strike_times)
        if not times:
            text = "None yet. Move the playhead and press Strike."
        else:
            listed = ", ".join(f"{t:.2f} s" for t in times[:8])
            more = f" and {len(times) - 8} more" if len(times) > 8 else ""
            text = f"{len(times)} placed: {listed}{more}"
        try:
            self.strike_label.setText(text)
        except RuntimeError:
            pass                            # the row was rebuilt meanwhile

    def _add_strike(self) -> None:
        item = self.base_item()
        if item is None:
            return
        t = round(float(self.owner.time), 3)
        if not any(abs(t - s) < 1e-3 for s in item.strike_times):
            item.strike_times = sorted([*item.strike_times, t])
        self._strikes_changed()

    def _remove_strike(self) -> None:
        from .fx3d import STRIKE_SECONDS
        item = self.base_item()
        if item is None or not item.strike_times:
            return
        t = float(self.owner.time)
        # The strike under the playhead: one starting here or still flashing.
        near = [s for s in item.strike_times if -0.05 <= t - s <= STRIKE_SECONDS]
        target = max(near) if near else min(item.strike_times, key=lambda s: abs(s - t))
        item.strike_times = [s for s in item.strike_times if s != target]
        self._strikes_changed()

    def _clear_strikes(self) -> None:
        item = self.base_item()
        if item is not None:
            item.strike_times = []
            self._strikes_changed()

    def _strikes_changed(self) -> None:
        self._show_strikes()
        self.owner.strikes_changed()

    def _colour_row(self) -> None:
        swatch = QPushButton()
        swatch.setFixedSize(46, 22)
        swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        swatch.setToolTip("Pick the colour.")
        swatch.clicked.connect(self._pick_colour)
        row = _Row("Colour", swatch)
        row.swatch = swatch
        row.diamond.clicked.connect(lambda: self._toggle_key(("colour",)))
        self._rows["colour"] = row
        self.props_layout.addWidget(row)

    # -- reading current values ------------------------------------------------

    def shown_item(self):
        """The selected effect as it is at the playhead (keys applied)."""
        state = self.owner.effects_at(self.owner.time)
        return state.item(self._item_id) if self._item_id else None

    def _paths(self, fields) -> list[str]:
        item = self.base_item()
        paths = []
        for f in fields:
            value = getattr(item, f)
            if isinstance(value, (tuple, list)):
                paths += [f"{item.id}.{f}.{i}" for i in range(len(value))]
            else:
                paths.append(f"{item.id}.{f}")
        return paths

    def _key_state(self, fields) -> str:
        track, t = self.owner.effects_track, self.owner.time
        paths = self._paths(fields)
        if paths and all(track.is_keyed(t, p) for p in paths):
            return "keyed"
        if any(track.is_animated(p) for p in paths):
            return "animated"
        return "none"

    def sync(self) -> None:
        """Show the values and diamonds for the playhead's time."""
        self._sync_env()
        item = self.shown_item()
        if item is None:
            return
        for field, slider in self._sliders.items():
            if field in ("tilt", "heading"):
                continue
            if field == "size":
                slider.set_value(float(np.mean(item.size)))
            else:
                slider.set_value(float(getattr(item, field)))
            slider.diamond.set_state(self._key_state((field,)))
        if "position" in self._rows:
            self._rows["position"].diamond.set_state(self._key_state(("position", "rotation")))
        if "colour" in self._rows:
            r, g, b = (int(max(0.0, min(1.0, c)) * 255) for c in item.colour)
            self._rows["colour"].swatch.setStyleSheet(
                f"background: rgb({r},{g},{b}); border: 1px solid #3a4459; border-radius: 4px;")
            self._rows["colour"].diamond.set_state(self._key_state(("colour",)))
        if "direction" in self._rows:
            row = self._rows["direction"]
            name = direction_name(item.direction)
            row.combo.blockSignals(True)
            row.combo.setCurrentText(name)
            row.combo.blockSignals(False)
            tilt, heading = direction_to_angles(item.direction)
            self._sliders["tilt"].set_value(tilt)
            self._sliders["heading"].set_value(heading)
            for f in ("tilt", "heading"):
                self._sliders[f].setVisible(name == "Custom")
            row.diamond.set_state(self._key_state(("direction",)))

    # -- writing ---------------------------------------------------------------

    def set_value(self, field: str, value) -> None:
        """Change a property: on the base when unanimated, as a key when not."""
        base = self.base_item()
        if base is None:
            return
        if isinstance(getattr(base, field), bool):
            value = bool(value)
        elif isinstance(getattr(base, field), int):
            value = int(round(value))
        paths = self._paths((field,))
        track = self.owner.effects_track
        if any(track.is_animated(p) for p in paths):
            state = self.owner.effects_at(self.owner.time).clone()
            setattr(state.item(base.id), field, value)
            for p in paths:
                track.key_property(self.owner.time, p, state)
        else:
            setattr(base, field, value)
        if field == "name":
            self._refresh_list()
        self.owner.fx_changed()
        self.sync()

    def _set_size(self, value: float) -> None:
        item = self.shown_item()
        if item is None:
            return
        if kind_of(item) == "plane":
            cur = float(np.mean([item.size[0], item.size[2]])) or 1.0
            k = float(value) / cur
            new = (item.size[0] * k, item.size[1], item.size[2] * k)
        else:
            cur = float(np.mean(item.size)) or 1.0
            new = tuple(float(s) * float(value) / cur for s in item.size)
        self.set_value("size", new)

    def _toggle_key(self, fields) -> None:
        if self.base_item() is None:
            return
        track, t = self.owner.effects_track, self.owner.time
        paths = self._paths(fields)
        if paths and all(track.is_keyed(t, p) for p in paths):
            for p in paths:
                track.unkey_property(t, p)
        else:
            state = self.owner.effects_at(t).clone()
            for p in paths:
                track.key_property(t, p, state)
        self.owner.keys_changed()
        self.sync()

    def _direction_preset(self, name: str) -> None:
        if name in DIRECTIONS:
            self.set_value("direction", angles_to_direction(*DIRECTIONS[name]))
        else:
            self.sync()

    def _direction_angles(self) -> None:
        self.set_value("direction", angles_to_direction(self._sliders["tilt"].value.value(),
                                                        self._sliders["heading"].value.value()))

    def _pick_colour(self) -> None:
        item = self.shown_item()
        if item is None:
            return
        r, g, b = (int(max(0.0, min(1.0, c)) * 255) for c in item.colour)
        chosen = QColorDialog.getColor(QColor(r, g, b), self, "Effect colour")
        if chosen.isValid():
            self.set_value("colour", (chosen.redF(), chosen.greenF(), chosen.blueF()))

    def moved(self, item_id: str, shown) -> None:
        """The gizmo moved an effect in the 3D view (which shows `shown`)."""
        self.select(item_id)
        base = self.owner.effects.item(item_id)
        moved = shown.item(item_id) if shown is not None else None
        if base is None or moved is None or base is moved:
            self.sync()
            return
        track, t = self.owner.effects_track, self.owner.time
        for field in ("position", "rotation", "size"):
            paths = self._paths((field,)) if self._item_id == item_id else []
            if paths and any(track.is_animated(p) for p in paths):
                for p in paths:
                    track.key_property(t, p, shown)
            else:
                setattr(base, field, getattr(moved, field))
        self.owner.keys_changed()
        self.sync()

    def _duplicate(self) -> None:
        import copy
        from .effects3d import _id
        base = self.base_item()
        if base is None:
            return
        twin = copy.deepcopy(base)
        twin.id = _id()
        twin.name = f"{base.name} copy"
        x, y, z = base.position
        twin.position = (x + 0.3, y, z)
        for attr in ("volumes", "emitters", "planes", "strikes"):
            group = getattr(self.owner.effects, attr)
            if any(i.id == base.id for i in group):
                group.append(twin)
        self._added(twin)

    def _remove(self) -> None:
        base = self.base_item()
        if base is None:
            return
        fx = self.owner.effects
        order = [i.id for i in fx.items()]
        index = order.index(base.id)
        for attr in ("volumes", "emitters", "planes", "strikes"):
            setattr(fx, attr, [i for i in getattr(fx, attr) if i.id != base.id])
        # Select the neighbour, so removing several in a row is one click each.
        rest = [i.id for i in fx.items()]
        self._item_id = rest[min(index, len(rest) - 1)] if rest else None
        self._refresh_list()
        self.owner.fx_changed()

    # -- scene-wide (environment) properties -----------------------------------

    def _env_path(self, field: str) -> str:
        from .effects3d import ENVIRONMENT
        return f"{ENVIRONMENT}.{field}"

    def _sync_env(self) -> None:
        lighting = self.owner.effects_at(self.owner.time).lighting
        track, t = self.owner.effects_track, self.owner.time
        for field, slider in self._env_sliders.items():
            slider.set_value(float(getattr(lighting, field)))
            path = self._env_path(field)
            slider.diamond.set_state("keyed" if track.is_keyed(t, path) else
                                     "animated" if track.is_animated(path) else "none")

    def set_env(self, field: str, value: float) -> None:
        track, t = self.owner.effects_track, self.owner.time
        path = self._env_path(field)
        if track.is_animated(path):
            state = self.owner.effects_at(t).clone()
            setattr(state.lighting, field, float(value))
            track.key_property(t, path, state)
        else:
            setattr(self.owner.effects.lighting, field, float(value))
        self.owner.fx_changed()
        self._sync_env()

    def _toggle_env_key(self, field: str) -> None:
        track, t = self.owner.effects_track, self.owner.time
        path = self._env_path(field)
        if track.is_keyed(t, path):
            track.unkey_property(t, path)
        else:
            track.key_property(t, path, self.owner.effects_at(t))
        self.owner.keys_changed()
        self._sync_env()

    def _key_all(self) -> None:
        self.owner.effects_track.add(self.owner.time, self.owner.effects_at(self.owner.time))
        self.owner.keys_changed()
        self.sync()

    def set_enabled(self, on: bool) -> None:
        self.add_button.setEnabled(on)
        self.key_all.setEnabled(on)
