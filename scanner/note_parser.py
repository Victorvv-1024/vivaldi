from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sahi.prediction import ObjectPrediction


NOTEHEAD_CLASSES = {
    "noteheadBlackOnLine",
    "noteheadBlackOnLineSmall",
    "noteheadBlackInSpace",
    "noteheadBlackInSpaceSmall",
    "noteheadHalfOnLine",
    "noteheadHalfOnLineSmall",
    "noteheadHalfInSpace",
    "noteheadHalfInSpaceSmall",
    "noteheadWholeOnLine",
    "noteheadWholeOnLineSmall",
    "noteheadWholeInSpace",
    "noteheadWholeInSpaceSmall",
    "noteheadDoubleWholeOnLine",
    "noteheadDoubleWholeOnLineSmall",
    "noteheadDoubleWholeInSpace",
    "noteheadDoubleWholeInSpaceSmall",
    "noteheadFull",
    "noteheadFullSmall",
}

CLEF_REFERENCE = {
    "clefG": ("E", 4),  # bottom line
    "clefF": ("G", 2),
    "clefCAlto": ("F", 3),
    "clefCTenor": ("D", 3),
}

ACCIDENTAL_MAP = {
    "accidentalSharp": "#",
    "accidentalSharpSmall": "#",
    "accidentalFlat": "b",
    "accidentalFlatSmall": "b",
    "accidentalNatural": "n",
    "accidentalNaturalSmall": "n",
    "accidentalDoubleSharp": "x",
    "accidentalDoubleFlat": "bb",
}

NOTE_LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
LETTER_TO_INDEX: Dict[str, int] = {letter: idx for idx, letter in enumerate(NOTE_LETTERS)}


@dataclass
class NoteEvent:
    name: str
    octave: int
    accidental: str
    staff_index: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    symbol_category: str

    @property
    def label(self) -> str:
        if self.accidental and self.accidental != "n":
            return f"{self.name}{self.accidental}{self.octave}"
        return f"{self.name}{self.octave}"


@dataclass
class StaffRegion:
    bbox: Tuple[float, float, float, float]
    idx: int
    clef: str
    base_diatonic: int
    half_spacing: float
    bottom: float
    accidentals: List[ObjectPrediction]

    def pitch_from_step(self, step: int) -> Tuple[str, int]:
        number = self.base_diatonic + step
        octave = number // 7
        letter = NOTE_LETTERS[number % 7]
        return letter, octave


def infer_notes(predictions: List[ObjectPrediction]) -> List[NoteEvent]:
    staffs = _build_staff_regions(predictions)
    if not staffs:
        return []

    note_events: List[NoteEvent] = []
    for pred in predictions:
        name = pred.category.name
        if name not in NOTEHEAD_CLASSES:
            continue
        staff = _find_staff_for_prediction(pred, staffs)
        if staff is None:
            continue
        x1, y1, x2, y2 = pred.bbox.to_xyxy()
        center_y = (y1 + y2) / 2.0
        center_x = (x1 + x2) / 2.0
        if staff.half_spacing == 0:
            continue
        step = int(round((staff.bottom - center_y) / staff.half_spacing))
        note_letter, octave = staff.pitch_from_step(step)
        accidental = _match_accidental(center_x, center_y, staff)
        note_events.append(
            NoteEvent(
                name=note_letter,
                octave=octave,
                accidental=accidental,
                staff_index=staff.idx,
                bbox=(x1, y1, x2, y2),
                confidence=pred.score.value,
                symbol_category=name,
            )
        )

    return note_events


def _build_staff_regions(predictions: List[ObjectPrediction]) -> List[StaffRegion]:
    staff_preds = [p for p in predictions if p.category.name == "staff"]
    staff_preds.sort(key=lambda p: p.bbox.to_xyxy()[1])
    regions: List[StaffRegion] = []
    for idx, staff_pred in enumerate(staff_preds):
        x1, y1, x2, y2 = staff_pred.bbox.to_xyxy()
        height = max(y2 - y1, 1.0)
        half_spacing = height / 8.0
        clef_name = _detect_clef_for_staff(staff_pred, predictions)
        base_letter, base_octave = CLEF_REFERENCE.get(clef_name, CLEF_REFERENCE["clefG"])
        base_number = int(base_octave) * 7 + LETTER_TO_INDEX[base_letter]
        accidentals = _collect_staff_accidentals(staff_pred, predictions)
        regions.append(
            StaffRegion(
                bbox=(x1, y1, x2, y2),
                idx=idx,
                clef=clef_name,
                base_diatonic=base_number,
                half_spacing=half_spacing,
                bottom=y2,
                accidentals=accidentals,
            )
        )
    return regions


def _detect_clef_for_staff(staff_pred: ObjectPrediction, predictions: List[ObjectPrediction]) -> str:
    x1, y1, x2, y2 = staff_pred.bbox.to_xyxy()
    candidates = []
    for pred in predictions:
        name = pred.category.name
        if not name.startswith("clef"):
            continue
        px1, py1, px2, py2 = pred.bbox.to_xyxy()
        if py2 < y1 or py1 > y2:
            continue
        if px1 > x2:
            continue
        candidates.append((px1, name))
    if not candidates:
        return "clefG"
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _collect_staff_accidentals(staff_pred: ObjectPrediction, predictions: List[ObjectPrediction]) -> List[ObjectPrediction]:
    x1, y1, x2, y2 = staff_pred.bbox.to_xyxy()
    accs = []
    for pred in predictions:
        name = pred.category.name
        if name not in ACCIDENTAL_MAP:
            continue
        px1, py1, px2, py2 = pred.bbox.to_xyxy()
        if py2 < y1 or py1 > y2:
            continue
        accs.append(pred)
    return accs


def _find_staff_for_prediction(pred: ObjectPrediction, staffs: List[StaffRegion]) -> Optional[StaffRegion]:
    x1, y1, x2, y2 = pred.bbox.to_xyxy()
    center_y = (y1 + y2) / 2.0
    overlaps = [s for s in staffs if s.bbox[1] <= center_y <= s.bbox[3]]
    if not overlaps:
        # fallback: choose closest
        overlaps = sorted(staffs, key=lambda s: abs(((s.bbox[1] + s.bbox[3]) / 2.0) - center_y))
    return overlaps[0] if overlaps else None


def _match_accidental(center_x: float, center_y: float, staff: StaffRegion) -> str:
    tolerance = staff.half_spacing * 3
    best: Optional[Tuple[float, str]] = None
    for acc in staff.accidentals:
        if acc.category.name not in ACCIDENTAL_MAP:
            continue
        ax1, ay1, ax2, ay2 = acc.bbox.to_xyxy()
        if not (ay1 - staff.half_spacing <= center_y <= ay2 + staff.half_spacing):
            continue
        distance = center_x - ax2
        if distance < 0 or distance > tolerance:
            continue
        if best is None or distance < best[0]:
            best = (distance, ACCIDENTAL_MAP[acc.category.name])
    return best[1] if best else ""
