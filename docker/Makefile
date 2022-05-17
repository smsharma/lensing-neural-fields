default: build release
build:
	docker build -t lensing-nerf .
clean-build:
	docker build -t lensing-nerf . --no-cache=true
run:
	docker run model-free
release:
	docker tag model-free improbableailab/lensing-nerf
	docker push improbableailab/lensing-nerf
