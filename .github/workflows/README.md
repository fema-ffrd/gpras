### Description of workflows

### ci.yaml

This is triggered each time a pull request is submitted or code is pushed to either `main` or `dev`. A pip wheel is then built for python versions 3.10, 3.11, and 3.12. Finally, ruff is used to check formatting.

### dev-push.yaml

On pushes to `dev`, trigger build of the dev docker container and push to GitHub container registry. (calls docker-build.yaml)

### main-push.yaml

On pushes to `main`, trigger build of the docker container and push to GitHub container registry. (calls docker-build.yaml)

### docker-build.yaml

Triggered by pushes to `main` and `dev`.  Build a docker container and push to GitHub container registry.

### pr-checks.yaml

Triggered by pull requests on `main` and `dev`.  Build a docker container but don't push to GitHub container registry.

### release.yaml

Build a pip wheel for the repository, make a version release doc, and upload source distribution to the release.
