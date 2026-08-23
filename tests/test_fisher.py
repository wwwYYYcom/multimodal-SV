from pathlib import Path

from mmsv.data.fisher import build_manifest, read_manifest


def test_build_manifest(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    transcripts = tmp_path / "trans"
    audio.mkdir()
    transcripts.mkdir()
    (audio / "fe_03_00001.sph").write_bytes(b"placeholder")
    (transcripts / "fe_03_00001.txt").write_text(
        "# header\n0.00 1.50 A: hello\n2.00 2.40 B: too short\n3.00 5.00 B: world\n",
        encoding="utf-8",
    )
    calldata = tmp_path / "calldata.tbl"
    calldata.write_text(
        "CALL_ID,APIN,BPIN\n00001,1001,1002\n", encoding="utf-8"
    )
    output = tmp_path / "manifest.csv"
    audit = build_manifest(audio, transcripts, calldata, output, min_duration=1.0)
    rows = read_manifest(output)
    assert audit["utterances"] == 2
    assert [row["speaker_id"] for row in rows] == ["1001", "1002"]
    assert rows[0]["utt_id"] == "fe_03_00001_A_0001"

