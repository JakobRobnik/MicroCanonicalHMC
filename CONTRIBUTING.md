# How to contribute to MCHMC

## Bugs

### Did you find a bug?

**Ensure the bug was not already reported** by searching on GitHub under
[Issues](https://github.com/JakobRobnik/MicroCanonicalHMC/issues/new). If you're unable to find an
open issue addressing the problem, [open a new
one](https://github.com/minaskar/pocomc/issues/new). Be sure to include a **title
and clear description**, as much relevant information as possible, and the
simplest possible **code sample** demonstrating the expected behavior that is
not occurring.

### Did you write a patch that fixes a bug?

Open a new GitHub pull request with the patch. Ensure the PR description
clearly describes the problem and solution. Include the relevant issue number
if applicable.

## Do you intend to add a new feature or change an existing one?

First, [open a new issue](https://github.com/JakobRobnik/MicroCanonicalHMC/issues/new) and
clearly describe your idea. We will then let you know if what you suggest 
aligns with the vision of MCHMC. This way you might avoid doing unnecessary
work and might even find some help from other people.

### Contribution Workflow

1. Checkout a new branch.
2. Run `make set-bench` to determine the speed of the current version of the code on your computer.
3. Add your feature on a new branch
4. Run `make test` to run tests locally.
5. Run `make compare-bench` to see if your changes slowed the code.
6. Push the branch and open a pull request. Assign a developer.
