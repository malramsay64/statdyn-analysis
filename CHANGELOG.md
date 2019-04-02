<a name="unreleased"></a>
## [Unreleased]


<a name="v0.7.3"></a>
## [v0.7.3] - 2019-04-02
### Chore
- Build documentation on travis-ci
- Pin numpy version
- Add python 3.7 to conda_build_config
- Fix warnings about dependency specifications in meta.yaml
- Use local install of pylint for pre-commit
- Add makefile rule for building docs
- Include requirements for building documentation
- Fix indentation of .pre-commit-config.yaml

### Docs
- Improve display of dynamics class in docs
- Update modules in project documentation

### Feat
- Use pip for testing on travis

### Fix
- Pylint issues raised in CI
- Include black in dev requirements of setup.py
- Use the xenial distribution on travis
- typo in docs_requires for setup.py
- Call relative_distances function correctly in tests
- Fix unused imports and whitespace issues
- Version 1.0 of freud changes the interface for neighbour distances


<a name="v0.7.2"></a>
## [v0.7.2] - 2019-03-19
### Chore
- Set specific pylint version in pre-commit config
- Include pre_commit in conda environment
- Add pylint to pre-commit checks
- Ignore interactive config in coverage reports ([#56](https://github.com/malramsay64/statdyn-analysis/issues/56))
- Update pre-commit configuration
- Remove unused code from interactive_config
- Update play/pause button to work with bokeh 1.0
- Update scikit-learn to 0.20 in environment.yml

### Doc
- Add docstring to interactive configuration

### Feat
- Output timestep as a comma separated integer
- Improve debug logging of interactive_config
- Update interactive slider range on loading file
- Shown variables reflect the available variables
- Implement verbose flag for figure subcommand

### Fix
- Fix use of formatting in raising of exception.
- Variables not defined and checking for None
- Load the dump files instead of all gsd files

### Fix
- No infinite loops in reading input file

### Sty
- Change name of the interacctive_logger


<a name="v0.7.1"></a>
## [v0.7.1] - 2019-03-06
### Fix
- Disable calculation of displacement across images

### Sty
- Disable unused argument for translational_displacement
- Sort imports and format with black
- Spelling mistakes in initial

### Test
- Increase max value of displacement
- Clean up dynamics tests using HoomdFrame


<a name="v0.7.0"></a>
## [v0.7.0] - 2019-03-04
### Chore
- Remove nonexistant models files from MANIFEST.in
- Remove empty .gitmodules file
- Remove scikit learn models from package
- Update to latest scikit learn ([#58](https://github.com/malramsay64/statdyn-analysis/issues/58))
- Apply pep8 naming conventions to dynamics module ([#65](https://github.com/malramsay64/statdyn-analysis/issues/65))

### Feat
- Handle pressure and temperature at any position in filename ([#68](https://github.com/malramsay64/statdyn-analysis/issues/68))
- Specify number of threads for parallel processing function
- Log quantities of interest when processing files ([#45](https://github.com/malramsay64/statdyn-analysis/issues/45))
- Use INFO log level for cli status update ([#72](https://github.com/malramsay64/statdyn-analysis/issues/72))
- Compute dynamics keeping track of images ([#71](https://github.com/malramsay64/statdyn-analysis/issues/71))
- Implement image in the Frame classes
- Support images in computing dynamics
- Allow displacement over multiple cells ([#71](https://github.com/malramsay64/statdyn-analysis/issues/71))
- Ability to specify custom models on command line
- Add keyframe-interval flag to cli

### Fix
- Always return a float from structural_relaxation
- Remove machine learning models from tests

### Sty
- Satiate pylint

### Style
- Format tests using black


<a name="v0.6.10"></a>
## [v0.6.10] - 2019-02-28
### Chore
- Add conda-forge repo to travis for build phase

### Feat
- Generate changelog using git-chglog

### Fix
- Handle diffusion constant where there is no relaxation

### Style
- Allow any ordering of arguments in a conditional


<a name="v0.6.9"></a>
## [v0.6.9] - 2019-02-25
### Chore
- closer specification of version numbers
- Run updated black on source files
- fix environment activation on travis
- Remove the conda environment lock mechanism
- The results from Rowan appear to have changed in an updates
- Use the pypi name for freud in setup.py

### Fix
- Use appropriate mypy type hinting for interactive_config
- Handle errors when running with no crystal value
- Include xy, xz, and yz components to box in lammps frame
- Remove error messages about Box
- Align simulation steps with analysis steps
- Support thermodynamics files in get_filename_vars function

### Style
- Format interactive_config using black


<a name="v0.6.8"></a>
## [v0.6.8] - 2018-10-16

<a name="v0.6.7"></a>
## [v0.6.7] - 2018-10-16

<a name="v0.6.6"></a>
## [v0.6.6] - 2018-09-25

<a name="v0.6.5"></a>
## [v0.6.5] - 2018-09-23

<a name="v0.6.4"></a>
## [v0.6.4] - 2018-09-21

<a name="v0.6.3"></a>
## [v0.6.3] - 2018-09-21

<a name="v0.6.2"></a>
## [v0.6.2] - 2018-09-13

<a name="v0.6.1"></a>
## [v0.6.1] - 2018-09-12

<a name="v0.6.0"></a>
## [v0.6.0] - 2018-09-10

<a name="v0.5.6"></a>
## [v0.5.6] - 2018-08-03

<a name="v0.5.5"></a>
## [v0.5.5] - 2018-07-27

<a name="v0.5.4"></a>
## [v0.5.4] - 2018-06-28

<a name="v0.5.3"></a>
## [v0.5.3] - 2018-06-13

<a name="v0.5.2"></a>
## [v0.5.2] - 2018-06-10

<a name="v0.5.1"></a>
## [v0.5.1] - 2018-06-08

<a name="v0.5.0"></a>
## [v0.5.0] - 2018-06-08

<a name="v0.4.12"></a>
## [v0.4.12] - 2018-03-31
### Bugfix
- variables from filename lost in rebase
- make gsdFrame and lammpsFrame a class


<a name="v0.4.11"></a>
## [v0.4.11] - 2018-03-21

<a name="v0.4.10"></a>
## [v0.4.10] - 2018-03-21

<a name="v0.4.9"></a>
## [v0.4.9] - 2018-02-23

<a name="v0.4.8"></a>
## [v0.4.8] - 2018-02-23
### Reverts
- Formatting - spaces around kwargs with type annotations


<a name="v0.4.7"></a>
## [v0.4.7] - 2018-02-23

<a name="v0.4.6"></a>
## [v0.4.6] - 2018-02-07

<a name="0.4.5"></a>
## [0.4.5] - 2018-01-03

<a name="v0.4.4.post6"></a>
## [v0.4.4.post6] - 2018-01-02

<a name="v0.4.4.post5"></a>
## [v0.4.4.post5] - 2018-01-02

<a name="0.4.4.post4"></a>
## [0.4.4.post4] - 2018-01-02

<a name="0.4.4.post3"></a>
## [0.4.4.post3] - 2018-01-02

<a name="0.4.4.post2"></a>
## [0.4.4.post2] - 2018-01-02

<a name="0.4.4.post1"></a>
## [0.4.4.post1] - 2018-01-02

<a name="v0.4.4"></a>
## [v0.4.4] - 2018-01-02

<a name="0.4.3"></a>
## [0.4.3] - 2017-12-19

<a name="0.4.2"></a>
## [0.4.2] - 2017-12-14

<a name="0.4.1"></a>
## [0.4.1] - 2017-12-13

<a name="0.4.0"></a>
## [0.4.0] - 2017-12-07

<a name="0.3.18"></a>
## [0.3.18] - 2017-12-06

<a name="0.3.17"></a>
## [0.3.17] - 2017-11-29

<a name="0.3.16"></a>
## [0.3.16] - 2017-11-21

<a name="0.3.15"></a>
## [0.3.15] - 2017-11-21

<a name="0.3.14"></a>
## [0.3.14] - 2017-11-21

<a name="0.3.13"></a>
## [0.3.13] - 2017-11-10

<a name="0.3.12"></a>
## [0.3.12] - 2017-11-08

<a name="0.3.10post12"></a>
## [0.3.10post12] - 2017-11-06

<a name="0.3.11"></a>
## [0.3.11] - 2017-11-06

<a name="0.3.10post11"></a>
## [0.3.10post11] - 2017-11-06

<a name="0.3.10post9"></a>
## [0.3.10post9] - 2017-11-06

<a name="0.3.10post5"></a>
## [0.3.10post5] - 2017-11-06

<a name="0.3.10post2"></a>
## [0.3.10post2] - 2017-11-06

<a name="0.3.10post1"></a>
## [0.3.10post1] - 2017-11-06

<a name="0.3.10"></a>
## [0.3.10] - 2017-11-06

<a name="0.3.9"></a>
## [0.3.9] - 2017-11-06

<a name="0.3.8"></a>
## [0.3.8] - 2017-11-02

<a name="0.3.7"></a>
## [0.3.7] - 2017-10-28

<a name="0.3.6"></a>
## [0.3.6] - 2017-10-18

<a name="0.3.5"></a>
## [0.3.5] - 2017-10-17

<a name="0.3.4"></a>
## [0.3.4] - 2017-10-09

<a name="0.3.3"></a>
## [0.3.3] - 2017-10-05

<a name="0.3.2"></a>
## [0.3.2] - 2017-10-01

<a name="0.3.1"></a>
## [0.3.1] - 2017-09-29

<a name="0.3.0"></a>
## [0.3.0] - 2017-09-22

<a name="0.2.9"></a>
## [0.2.9] - 2017-09-19

<a name="0.2.8"></a>
## [0.2.8] - 2017-09-13

<a name="0.2.7"></a>
## [0.2.7] - 2017-09-12
### Bugfix
- Compare current step with frame step


<a name="0.2.6"></a>
## [0.2.6] - 2017-09-09

<a name="0.2.5"></a>
## [0.2.5] - 2017-09-08

<a name="0.2.4"></a>
## [0.2.4] - 2017-09-07

<a name="0.2.3"></a>
## [0.2.3] - 2017-09-06
### Bugfix
- No excess calls to the _enqueue function


<a name="0.2.2"></a>
## [0.2.2] - 2017-09-01

<a name="0.2.1"></a>
## [0.2.1] - 2017-08-28

<a name="0.2.0"></a>
## [0.2.0] - 2017-08-26

<a name="0.1.9"></a>
## [0.1.9] - 2017-08-23

<a name="0.1.8"></a>
## [0.1.8] - 2017-08-20

<a name="0.1.7"></a>
## [0.1.7] - 2017-08-14

<a name="0.1.6"></a>
## [0.1.6] - 2017-08-14

<a name="0.1.5"></a>
## [0.1.5] - 2017-08-11

<a name="0.1.4"></a>
## [0.1.4] - 2017-08-01

<a name="0.1.3"></a>
## [0.1.3] - 2017-07-24

<a name="0.1.2"></a>
## [0.1.2] - 2017-07-14

<a name="0.1.1"></a>
## [0.1.1] - 2017-07-13

<a name="0.1.0"></a>
## [0.1.0] - 2017-07-10

<a name="0.0.8"></a>
## 0.0.8 - 2016-09-10

[Unreleased]: https://github.com/malramsay64/statdyn-analysis/compare/v0.7.3...HEAD
[v0.7.3]: https://github.com/malramsay64/statdyn-analysis/compare/v0.7.2...v0.7.3
[v0.7.2]: https://github.com/malramsay64/statdyn-analysis/compare/v0.7.1...v0.7.2
[v0.7.1]: https://github.com/malramsay64/statdyn-analysis/compare/v0.7.0...v0.7.1
[v0.7.0]: https://github.com/malramsay64/statdyn-analysis/compare/v0.6.10...v0.7.0
[v0.6.10]: https://github.com/malramsay64/statdyn-analysis/compare/v0.6.9...v0.6.10
[v0.6.9]: https://github.com/malramsay64/statdyn-analysis/compare/v0.6.8...v0.6.9
[v0.6.8]: https://github.com/malramsay64/statdyn-analysis/compare/v0.6.7...v0.6.8
[v0.6.7]: https://github.com/malramsay64/statdyn-analysis/compare/v0.6.6...v0.6.7
[v0.6.6]: https://github.com/malramsay64/statdyn-analysis/compare/v0.6.5...v0.6.6
[v0.6.5]: https://github.com/malramsay64/statdyn-analysis/compare/v0.6.4...v0.6.5
[v0.6.4]: https://github.com/malramsay64/statdyn-analysis/compare/v0.6.3...v0.6.4
[v0.6.3]: https://github.com/malramsay64/statdyn-analysis/compare/v0.6.2...v0.6.3
[v0.6.2]: https://github.com/malramsay64/statdyn-analysis/compare/v0.6.1...v0.6.2
[v0.6.1]: https://github.com/malramsay64/statdyn-analysis/compare/v0.6.0...v0.6.1
[v0.6.0]: https://github.com/malramsay64/statdyn-analysis/compare/v0.5.6...v0.6.0
[v0.5.6]: https://github.com/malramsay64/statdyn-analysis/compare/v0.5.5...v0.5.6
[v0.5.5]: https://github.com/malramsay64/statdyn-analysis/compare/v0.5.4...v0.5.5
[v0.5.4]: https://github.com/malramsay64/statdyn-analysis/compare/v0.5.3...v0.5.4
[v0.5.3]: https://github.com/malramsay64/statdyn-analysis/compare/v0.5.2...v0.5.3
[v0.5.2]: https://github.com/malramsay64/statdyn-analysis/compare/v0.5.1...v0.5.2
[v0.5.1]: https://github.com/malramsay64/statdyn-analysis/compare/v0.5.0...v0.5.1
[v0.5.0]: https://github.com/malramsay64/statdyn-analysis/compare/v0.4.12...v0.5.0
[v0.4.12]: https://github.com/malramsay64/statdyn-analysis/compare/v0.4.11...v0.4.12
[v0.4.11]: https://github.com/malramsay64/statdyn-analysis/compare/v0.4.10...v0.4.11
[v0.4.10]: https://github.com/malramsay64/statdyn-analysis/compare/v0.4.9...v0.4.10
[v0.4.9]: https://github.com/malramsay64/statdyn-analysis/compare/v0.4.8...v0.4.9
[v0.4.8]: https://github.com/malramsay64/statdyn-analysis/compare/v0.4.7...v0.4.8
[v0.4.7]: https://github.com/malramsay64/statdyn-analysis/compare/v0.4.6...v0.4.7
[v0.4.6]: https://github.com/malramsay64/statdyn-analysis/compare/0.4.5...v0.4.6
[0.4.5]: https://github.com/malramsay64/statdyn-analysis/compare/v0.4.4.post6...0.4.5
[v0.4.4.post6]: https://github.com/malramsay64/statdyn-analysis/compare/v0.4.4.post5...v0.4.4.post6
[v0.4.4.post5]: https://github.com/malramsay64/statdyn-analysis/compare/0.4.4.post4...v0.4.4.post5
[0.4.4.post4]: https://github.com/malramsay64/statdyn-analysis/compare/0.4.4.post3...0.4.4.post4
[0.4.4.post3]: https://github.com/malramsay64/statdyn-analysis/compare/0.4.4.post2...0.4.4.post3
[0.4.4.post2]: https://github.com/malramsay64/statdyn-analysis/compare/0.4.4.post1...0.4.4.post2
[0.4.4.post1]: https://github.com/malramsay64/statdyn-analysis/compare/v0.4.4...0.4.4.post1
[v0.4.4]: https://github.com/malramsay64/statdyn-analysis/compare/0.4.3...v0.4.4
[0.4.3]: https://github.com/malramsay64/statdyn-analysis/compare/0.4.2...0.4.3
[0.4.2]: https://github.com/malramsay64/statdyn-analysis/compare/0.4.1...0.4.2
[0.4.1]: https://github.com/malramsay64/statdyn-analysis/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.18...0.4.0
[0.3.18]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.17...0.3.18
[0.3.17]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.16...0.3.17
[0.3.16]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.15...0.3.16
[0.3.15]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.14...0.3.15
[0.3.14]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.13...0.3.14
[0.3.13]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.12...0.3.13
[0.3.12]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.10post12...0.3.12
[0.3.10post12]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.11...0.3.10post12
[0.3.11]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.10post11...0.3.11
[0.3.10post11]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.10post9...0.3.10post11
[0.3.10post9]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.10post5...0.3.10post9
[0.3.10post5]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.10post2...0.3.10post5
[0.3.10post2]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.10post1...0.3.10post2
[0.3.10post1]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.10...0.3.10post1
[0.3.10]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.9...0.3.10
[0.3.9]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.8...0.3.9
[0.3.8]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.7...0.3.8
[0.3.7]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.6...0.3.7
[0.3.6]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.5...0.3.6
[0.3.5]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.4...0.3.5
[0.3.4]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.3...0.3.4
[0.3.3]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.2...0.3.3
[0.3.2]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/malramsay64/statdyn-analysis/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/malramsay64/statdyn-analysis/compare/0.2.9...0.3.0
[0.2.9]: https://github.com/malramsay64/statdyn-analysis/compare/0.2.8...0.2.9
[0.2.8]: https://github.com/malramsay64/statdyn-analysis/compare/0.2.7...0.2.8
[0.2.7]: https://github.com/malramsay64/statdyn-analysis/compare/0.2.6...0.2.7
[0.2.6]: https://github.com/malramsay64/statdyn-analysis/compare/0.2.5...0.2.6
[0.2.5]: https://github.com/malramsay64/statdyn-analysis/compare/0.2.4...0.2.5
[0.2.4]: https://github.com/malramsay64/statdyn-analysis/compare/0.2.3...0.2.4
[0.2.3]: https://github.com/malramsay64/statdyn-analysis/compare/0.2.2...0.2.3
[0.2.2]: https://github.com/malramsay64/statdyn-analysis/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/malramsay64/statdyn-analysis/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/malramsay64/statdyn-analysis/compare/0.1.9...0.2.0
[0.1.9]: https://github.com/malramsay64/statdyn-analysis/compare/0.1.8...0.1.9
[0.1.8]: https://github.com/malramsay64/statdyn-analysis/compare/0.1.7...0.1.8
[0.1.7]: https://github.com/malramsay64/statdyn-analysis/compare/0.1.6...0.1.7
[0.1.6]: https://github.com/malramsay64/statdyn-analysis/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/malramsay64/statdyn-analysis/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/malramsay64/statdyn-analysis/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/malramsay64/statdyn-analysis/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/malramsay64/statdyn-analysis/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/malramsay64/statdyn-analysis/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/malramsay64/statdyn-analysis/compare/0.0.8...0.1.0
