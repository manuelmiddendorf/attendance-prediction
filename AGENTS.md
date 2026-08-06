# AGENTS

## Project Philosophy

This repository documents an end-to-end machine learning project for
predicting attendance at yoga classes.

The repository is organized as a research project rather than a software
package.

- The notebooks tell the scientific story.
- The `src` package contains reusable implementation.
- Every design decision should prioritize readability,
  reproducibility, and maintainability over clever abstractions.

------------------------------------------------------------------------

## Repository Organization

### High-level structure

The repository should broadly follow this organization:

``` text
notebooks/
    01_data_understanding.ipynb
    02_preprocessing.ipynb
    03_feature_engineering.ipynb
    04_modeling.ipynb

src/
    data/
    data_understanding/
    preprocessing/
    feature_engineering/
    modeling/
```

The package structure should mirror the major project stages without
prescribing exact module names. Create modules only when they are
actually needed.

Every reusable operation must have exactly one canonical implementation.

Do not duplicate logic across notebook stages.

------------------------------------------------------------------------

## Notebooks vs. `src`

Before creating a function, always ask:

> Is this reusable implementation, or is it simply expressing an
> analytical observation?

### Keep code in notebooks

Keep code in notebooks when it

- communicates the analytical narrative,
- is a simple pandas expression,
- is used only once,
- directly displays or summarizes data,
- is easier to understand inline than through another abstraction.

Typical examples:

``` python
df.head()
df.info()
df.describe()
df.isna().sum()
df.duplicated().sum()
df["studio"].value_counts()
```

### Move code into `src`

Move code into `src` when it

- is reused,
- contains business logic,
- performs parsing or transformations,
- performs validation,
- computes reusable statistics,
- builds reusable plots,
- should be unit tested,
- would otherwise clutter the notebook.

Avoid trivial wrapper functions around obvious pandas operations.

Do not introduce abstractions until there is a concrete second use case.

Meaning is more important than line count.

------------------------------------------------------------------------

## Plotting

Reusable or publication-quality plots belong in a `plots.py` module.

Small exploratory plots may remain in notebooks.

------------------------------------------------------------------------

## Documentation

- Every public module should contain a module docstring.
- Every public function should contain a meaningful NumPy-style
  docstring.
- Comments should explain reasoning rather than obvious code.

------------------------------------------------------------------------

## Code Quality

- Use descriptive names.
- Add type hints to public functions.
- Keep functions focused.
- Preserve deterministic behaviour where appropriate.
- Follow the project's formatter and linter configuration.

------------------------------------------------------------------------

## Data Contracts

Each function should have a clear input and output contract.

Implement functions against the documented dataset schema and the documented
pipeline stage.

Do not add support for hypothetical input formats or malformed data unless
explicitly requested.

Unexpected inputs should fail clearly rather than being silently normalized,
repaired, or coerced into another format.

A function should perform only the transformations that belong to its
responsibility.

Examples:

- Data loading reads files without modifying their contents.
- Canonical representation changes only the representation, not the data.
- Data understanding observes the data without cleaning it.
- Cleaning and preprocessing are the stages where data modifications belong.

------------------------------------------------------------------------

## Prefer Simplicity

Prefer the simplest implementation that satisfies the documented requirements.

Do not add defensive programming, compatibility layers, fallback behavior, or
support for hypothetical edge cases unless they are part of the documented
dataset or explicitly requested.

Prefer built-in pandas functionality over custom helper functions whenever it
expresses the intended data contract directly.

Do not write parsing or conversion helpers if pandas already provides a clear,
equivalent operation.

------------------------------------------------------------------------

## Workflow

When implementing a notebook section:

- implement only the requested section,
- create only the functions required,
- do not implement future notebook sections,
- do not reorganize unrelated code,
- avoid placeholder modules.

When finished, briefly report

- what remained in the notebook,
- what moved into `src`,
- why those decisions were made.

------------------------------------------------------------------------

## README

The README should act as the abstract of the project.

It should

- explain the motivation,
- describe the methodology,
- explain the repository structure,
- explain reproducibility,
- avoid duplicating notebook details.

Write concise professional English and keep the README synchronized with
the repository.
