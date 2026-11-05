# Releasing PowerShard

The **CI and release** workflow runs CPU tests and checks the install ZIP for pushes to `main` and pull requests into `main`. A pushed stable tag `vX.Y.Z` pointing to a commit in `main` runs the same checks. After they pass, the workflow publishes the ZIP, SHA-256 checksums and the manifest to GitHub Releases with the built-in `GITHUB_TOKEN`. CUDA/NCCL and POWER9 tests still run separately on the target hardware.

1. Update `version` in `pyproject.toml` and `__version__` in `powershard/__init__.py`. Add the release notes under `## X.Y.Z` in `CHANGELOG.md`.
2. Commit the changes, get them into `main`, and wait for CI to pass. Tag the intended commit on `main` and push the tag (example: `v0.6.1`):

   ```bash
   git switch main
   git pull --ff-only origin main
   git tag -a v0.6.1 -m "PowerShard 0.6.1"
   git push origin refs/tags/v0.6.1
   ```

3. Open **Actions → CI and release** to follow the run. A transient failure can be retried with **Re-run failed jobs**. Once a release is published, leave its tag and files in place; ship corrections with a new version.

For a local preview, run `python scripts/build_release.py` with Python 3.11 or later. It packages the committed `HEAD` into `dist/`. Add `--tag vX.Y.Z` to build a specific local tag and check that it matches the package version.

# Как выпустить релиз

Пайплайн **CI and release** запускает CPU-тесты и проверяет установочный ZIP при пуше в `main` и при обновлении pull request в `main`. Тег `vX.Y.Z` на коммите из `main` запускает те же проверки. После их успеха GitHub публикует ZIP, суммы SHA-256 и манифест со встроенным `GITHUB_TOKEN`. CUDA/NCCL и POWER9 проверяются отдельно на целевом железе.

1. Обновите `version` в `pyproject.toml` и `__version__` в `powershard/__init__.py`. Добавьте описание выпуска под заголовком `## X.Y.Z` в `CHANGELOG.md`.
2. Закоммитьте изменения, добавьте их в `main` и дождитесь зелёного CI. Поставьте тег на нужный коммит в `main` и отправьте его на GitHub, как в примере выше.
3. Следите за запуском в **Actions → CI and release**. При временном сбое используйте **Re-run failed jobs**. Исправления к уже опубликованному выпуску оформляйте новой версией.

Для локальной проверки выполните `python scripts/build_release.py` с Python 3.11 или новее. Скрипт собирает закоммиченный `HEAD` в `dist/`. Параметр `--tag vX.Y.Z` позволяет собрать конкретный локальный тег и сверить его с версией пакета.
