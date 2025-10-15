
rjob submit --name=qwen72b-qa \
--gpu=2 --memory=400000 \
--cpu=40 \
--charged-group=mllm \
--private-machine=group \
--mount=gpfs://gpfs1/mllm:/mnt/shared-storage-user/mllm \
--image=registry.h.pjlab.org.cn/ailab/pytorch:2.7.0-cuda12.8.1-py3.12-ubuntu24.04 \
-P 16 \
--host-network=true \
-e DISTRIBUTED_JOB=true \
-- bash -ex /path/1_generate_qa/gen.sh