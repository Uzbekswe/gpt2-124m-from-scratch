# VESSL setup for a future GPT-2 run

This milestone creates local templates only. It does not configure the VESSL CLI, access
your VESSL account, or launch cloud compute.

## Before any future run

1. Sign in at <https://cloud.vessl.ai/>.
2. Configure the CLI locally, outside this repository:

   ```bash
   vessl configure
   ```

3. Verify the selected organization and project:

   ```bash
   vessl whoami
   vessl configure list
   ```

4. Discover the resources actually available to your account:

   ```bash
   vessl cluster list
   vessl resource list
   vessl image list
   ```

5. Inspect and fill the templates in `configs/vessl/` using only values returned by your
   own VESSL account.
6. Launch a run only after explicit approval:

   ```bash
   vessl run create -f PATH_TO_YAML
   ```

## Credentials and private Git access

VESSL credentials live outside this Git repository. Do not add organization names,
project names, credential names, access keys, or tokens to tracked files.

Because this GitHub repository is private, create or select the required VESSL Git
credential/integration in the VESSL cloud console, then put only its local VESSL
credential name into the template's `credential_name` placeholder. The repository URL
must remain a normal HTTPS URL without embedded credentials.

Never place a GitHub personal access token in YAML, Python code, committed `.env` files,
or documentation examples. The repository's `.gitignore` excludes local `.env` files and
VESSL output directories, but an ignored secret should still never be intentionally added
to project source files.

## Reproducible run templates

Each template imports this private repository into `/workspace/gpt2-124m` and uses a
`REPLACE_WITH_GIT_COMMIT_SHA` placeholder. Replace it with an immutable commit SHA, not a
branch name, so a future run can be traced to the exact source version that created its
artifacts.

The `export` mapping reserves `/workspace/gpt2-124m/artifacts` for future checkpoints,
metrics, generated samples, and logs. Its `vessl-artifact://` destination intentionally
contains no organization or project identifier; fill any account-specific export target
only after reviewing your VESSL account and receiving approval.

VESSL run YAML groups run metadata, resources, image, repository import, artifact export,
and commands outside the model implementation. See the official [Run YAML
reference](https://docs.vessl.ai/reference/yaml/run-yaml) for current field details.
