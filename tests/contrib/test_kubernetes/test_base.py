import os
import tempfile
import time

import numpy as np
import pytest
from kubernetes import config
from sklearn.datasets import load_iris

from mlem.config import project_config
from mlem.constants import PREDICT_METHOD_NAME
from mlem.contrib.docker.base import DockerDaemon, DockerEnv
from mlem.contrib.docker.helpers import build_model_image
from mlem.contrib.kubernetes.base import (
    K8sDeployment,
    K8sDeploymentState,
    K8sEnv,
)
from mlem.core.objects import DeployStatus

from .utils import Command


@pytest.fixture
def load_kube_config():
    config.load_kube_config(os.getenv("KUBECONFIG", default="~/.kube/config"))


@pytest.fixture
def k8s_deployment(model_meta_saved_single):
    return K8sDeployment(
        name="ml",
        model=model_meta_saved_single.make_link(),
        image_pull_policy="Never",
        service_type="LoadBalancer",
    )


@pytest.fixture
def docker_image(k8s_deployment):
    tmpdir = tempfile.mkdtemp()
    k8s_deployment.dump(os.path.join(tmpdir, "deploy"))
    return build_model_image(
        k8s_deployment.get_model(),
        k8s_deployment.image_name,
        k8s_deployment.server
        or project_config(
            k8s_deployment.loc.project if k8s_deployment.is_saved else None
        ).server,
        DockerEnv(
            daemon=DockerDaemon(host=os.getenv("DOCKER_HOST", default=""))
        ),
        force_overwrite=True,
    )


@pytest.fixture
def k8s_deployment_state(docker_image, model_meta_saved_single):
    return K8sDeploymentState(
        image=docker_image,
        model_hash=model_meta_saved_single.meta_hash(),
    )


@pytest.fixture
def k8s_env():
    return K8sEnv(
        daemon=DockerDaemon(host=os.getenv("DOCKER_HOST", default=""))
    )


def test_deploy(
    load_kube_config, k8s_deployment, k8s_deployment_state, k8s_env
):
    k8s_deployment.update_state(k8s_deployment_state)
    assert k8s_env.get_status(k8s_deployment) == DeployStatus.NOT_DEPLOYED
    k8s_env.deploy(k8s_deployment)
    k8s_deployment.wait_for_status(
        DeployStatus.RUNNING,
        allowed_intermediate=[DeployStatus.STARTING],
        timeout=10,
        times=5,
    )
    time.sleep(5)
    assert k8s_env.get_status(k8s_deployment) == DeployStatus.RUNNING
    k8s_env.remove(k8s_deployment)
    time.sleep(5)
    assert k8s_env.get_status(k8s_deployment) == DeployStatus.NOT_DEPLOYED
    with k8s_env.daemon.client() as client:
        k8s_deployment_state.image.delete(client, force=True)
    time.sleep(5)


def test_deployed_service(
    load_kube_config, k8s_deployment, k8s_deployment_state, k8s_env
):
    time.sleep(15)
    k8s_deployment.update_state(k8s_deployment_state)
    k8s_env.deploy(k8s_deployment)
    cmd = Command("minikube tunnel")
    cmd.run(timeout=20, shell=True)
    client = k8s_deployment.get_client()
    train, _ = load_iris(return_X_y=True)
    response = client.predict(data=train)
    assert np.array_equal(response, np.array([0] * 50 + [1] * 50 + [2] * 50))
