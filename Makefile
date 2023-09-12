test:
	cd src ; pipenv run pytest -rxXs -m "not speed" ../tests/

test_speed:
	cd src ; pipenv run pytest -rxXs -m "speed" \
	--benchmark-compare=slowrun \
	../tests/

# test_development:
# 	cd src ; pipenv run pytest \
# 	--log-cli-level INFO \
# 	-x \
# 	-rXxs \
# 	-m "not speed" ../tests/

test_development:
	cd src ; pipenv run pytest \
	--log-cli-level INFO \
	-x \
	-rXxs \
	-k TestDisputeTrees ../tests/


# --log-cli-level INFO \
# --log-cli-level DEBUG \
# -k "test_inconsistent_with_default"
# -rxXs -m "not speed" ../tests/

# test_development:
# 	cd src ; pipenv run pytest --log-cli-level INFO \
# 	--log-format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s" \
# 	-rxXPs --runxfail \
# 	-k scikit_learn \
# 	../tests/ 


check_python_version:
	which python

# check python version before running, e.g., if pipenv was activated
