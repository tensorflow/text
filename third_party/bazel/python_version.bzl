"""Repository rule that exposes the hermetic Python version to BUILD files.

Generates `@python_version_repo//:py_version.bzl` with the Python version
selected by the HERMETIC_PYTHON_VERSION environment variable and the matching
requirements lock file.
"""

_PY_VERSION_BZL = """
HERMETIC_PYTHON_VERSION = "{version}"
REQUIREMENTS = "{requirements}"
"""

def _python_version_repo_impl(repository_ctx):
    version = repository_ctx.getenv(
        "HERMETIC_PYTHON_VERSION",
        repository_ctx.attr.default_version,
    )
    requirements = repository_ctx.attr.requirements.get(version)
    if not requirements:
        fail("Unsupported HERMETIC_PYTHON_VERSION={}; supported versions: {}".format(
            version,
            ", ".join(sorted(repository_ctx.attr.requirements.keys())),
        ))
    repository_ctx.file("BUILD.bazel", "")
    repository_ctx.file(
        "py_version.bzl",
        _PY_VERSION_BZL.format(
            version = version,
            requirements = requirements,
        ),
    )

python_version_repo = repository_rule(
    implementation = _python_version_repo_impl,
    attrs = {
        "default_version": attr.string(default = "3.11"),
        "requirements": attr.string_dict(mandatory = True),
    },
)
