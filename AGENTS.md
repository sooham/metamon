# Environments
The package manager in python here is `uv`. The development machine is a macbook pro M4 Pro with 1 TB of storage. Currently there is no production environement.

# Storage limitations of development machine 
Before running any commands which will generate a lot of files, check if the computer has enough free storage.

# Surveying the pokemon datasets directory
The pokemon datasets is in $METAMON_CACHE_DIR . when you look at subfolders in that be mindful because
the folders contain millions of files, using common bash commands like `ls`, `find` etc. will time out.
if the user has given you the exact battle id or filename , use that smogtours-gen1ou-749168_Unrated_encore90411_vs_mindplate96156_02-23-2024_WIN.json 
if you want to pick random battles or replay files , use `ls -f` in combination with tools like `head` or `tail` and others which are only going to read so many inodes in the folder.

# Asking the user for help
If you run into a bug i.e bad environment setups, an error you can't resolve, ambiguous references and error traces, please ask the user to clarify with more information.

# Tests and updating tests
The metamon repo has pytests tests , they can be run with `make test`. Analyzing if the test suite needs an updatee is mandatory. If you make critical changes or breaking changes you are expected to also update the tests. Your new test cases should be simple, composable and respect module and class boundaries. End2end tests  are in `uv run pytest tests/test_e2e_smoke.py tests/test_e2e_output.py -v` can combine multiple modules and classes to achive good test coverage. Mocking is done with monkeypatch if necessary.

# Performance
You should write code which if necessary and at your own discrection and determination of performance and runtime based on input size, should use parallelism such as threading , pooling, multi-process code if necessary, be mindful of shared resources and that functions being called are thread safe.  Other common perfomance optimizations include using caching in memory, writing to files for faster processing and reading from them on the next run are also good practices.