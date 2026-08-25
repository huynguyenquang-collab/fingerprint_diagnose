def test_upstream_module_imports_on_supported_python():
    from fpdiag.upstream import clone_upstream, download_official_publish_log

    assert callable(clone_upstream)
    assert callable(download_official_publish_log)
