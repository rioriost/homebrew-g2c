import nox

nox.options.python = "3.11"
nox.options.default_venv_backend = "uv"


@nox.session(python=["3.11"], tags=["lint"])
def lint(session):
    session.install("ruff")
    session.run("uv", "run", "ruff", "check", "src/g2c/main.py")
    session.run("uv", "run", "ruff", "format", "src/g2c/main.py")


@nox.session(python=["3.11"], tags=["mypy"])
def mypy(session):
    session.install(".")
    session.install("mypy", "types-antlr4-python3-runtime")
    session.run("uv", "run", "mypy", "src/g2c/main.py", "--follow-imports=skip")


@nox.session(python=["3.11"], tags=["pytest"])
def pytest(session):
    session.install(".")
    session.install("pytest", "pytest-cov")
    test_files = ["test.py"]
    session.run(
        "uv",
        "run",
        "pytest",
        "--maxfail=1",
        "--cov=g2c",
        "--cov-report=term",
        *test_files,
    )
