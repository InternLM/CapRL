

rjob submit --name=qwen3b-qa_answer \
--gpu=1 --memory=200000 \
--cpu=20 \
--charged-group=mllmexp_gpu \
--private-machine=group \
--mount=gpfs://gpfs1/mllm:/mnt/shared-storage-user/mllm \
--image=registry.h.pjlab.org.cn/ailab/pytorch:2.7.0-cuda12.8.1-py3.12-ubuntu24.04 \
-P 32 \
--host-network=true \
-e DISTRIBUTED_JOB=true \
-- bash -ex /path/3_answer_qa/launch.sh