"""setup/qblox-twin-requirements.txt is setup/requirements.txt without the IQM
client stack, plus qblox-scheduler. These tests keep it that way when
requirements.txt changes."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
QFW = ROOT / "setup" / "requirements.txt"
TWIN = ROOT / "setup" / "qblox-twin-requirements.txt"
COPY_MARKER = "# --- setup/requirements.txt without iqm-* ---"
ADDED_MARKER = "# --- added for the Qblox twin ---"


def _requirements(lines):
    out = []
    for line in lines:
        text = line.split("#", 1)[0].strip()
        if text:
            out.append(text)
    return out


def _is_iqm(requirement):
    return requirement.lower().startswith("iqm-")


def test_twin_requirements_follow_qfw_requirements():
    qfw = [r for r in _requirements(QFW.read_text().splitlines()) if not _is_iqm(r)]
    twin = TWIN.read_text().splitlines()
    copied = twin[twin.index(COPY_MARKER) + 1:twin.index(ADDED_MARKER)]

    assert _requirements(copied) == qfw, (
        "setup/qblox-twin-requirements.txt is out of step with "
        "setup/requirements.txt: copy its lines, without iqm-*, between the markers")


def test_twin_requirements_drop_iqm_and_pin_qblox_scheduler():
    twin = _requirements(TWIN.read_text().splitlines())
    added = _requirements(TWIN.read_text().split(ADDED_MARKER, 1)[1].splitlines())

    assert [r for r in _requirements(QFW.read_text().splitlines()) if _is_iqm(r)]
    assert not [r for r in twin if _is_iqm(r)]
    assert added == ["qblox-scheduler==1.0.1"]
