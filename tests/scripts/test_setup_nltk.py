from scripts import setup_nltk


def test_setup_nltk_downloads_wordnet(monkeypatch) -> None:
    downloaded: list[str] = []

    monkeypatch.setattr(setup_nltk, "_wordnet_is_installed", lambda: False)

    def fake_download(name: str) -> bool:
        downloaded.append(name)
        return True

    monkeypatch.setattr(setup_nltk.nltk, "download", fake_download)

    setup_nltk.main()

    assert downloaded == ["wordnet"]


def test_setup_nltk_skips_download_when_wordnet_exists(monkeypatch) -> None:
    monkeypatch.setattr(setup_nltk, "_wordnet_is_installed", lambda: True)
    monkeypatch.setattr(
        setup_nltk.nltk,
        "download",
        lambda name: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    setup_nltk.main()
