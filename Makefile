PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
.PHONY: setup models test demo evidence eval degrade bench deploy destroy

setup:
	python3 -m venv .venv && $(PIP) install -r requirements.lock
models:
	bash scripts/fetch_models.sh
test: models
	$(PY) -m pytest -q
demo: models
	$(PY) -m uvicorn holdsight.api:app --host 0.0.0.0 --port 8000
evidence:
	$(PY) scripts/cool_evidence.py --label $(LABEL) --cool-version "$(COOL_VERSION)"
eval:
	$(PY) -m eval.eyeblink8 --root $(EYEBLINK8)
degrade:
	$(PY) -m eval.degrade --clip $(CLIP)
bench:
	$(PY) -m eval.bench --label $(LABEL) --clips "$(CLIPS)" --usd-per-hour $(USD_PER_HOUR)
deploy:
	aws cloudformation deploy --stack-name $(STACK) --template-file infra/holdsight.yaml \
	  --capabilities CAPABILITY_IAM --region $(REGION) --parameter-overrides CoolAmiId=$(COOL_AMI)
destroy:
	aws cloudformation delete-stack --stack-name $(STACK) --region $(REGION)
