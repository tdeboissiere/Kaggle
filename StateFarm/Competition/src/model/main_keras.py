import sys
import os
import train
from dotenv import load_dotenv, find_dotenv
sys.path.append("../utils")


def launch_training(model_name, **kwargs):

    # Launch training
    if "adversarial" in model_name:
        train.train_adversarial(model_name, **kwargs)
    else:
        train.cross_validate_inmemory(model_name, **d_params)


def launch_submission(model_name, **kwargs):

    train.make_submission_CV(model_name, **kwargs)


if __name__ == "__main__":

    # Load env variables in (in .env file at the root of the project)
    load_dotenv(find_dotenv())

    # Load env variables
    data_dir = os.path.expanduser(os.environ.get("DATA_DIR"))
    model_dir = os.path.expanduser(os.environ.get("MODEL_DIR"))
    external_dir = os.path.expanduser(os.environ.get("EXTERNAL_DIR"))

    data_file = "fullframe_VGG_train.h5"
    data_file = os.path.join(data_dir, data_file)

    semi_super_file = "fullframe_VGG_test.h5"
    semi_super_file = os.path.join(data_dir, semi_super_file)

    pretr_weights_file = "vgg16_weights.h5"
    pretr_weights_file = os.path.join(external_dir, pretr_weights_file)

    tr_weights_file = ""
    tr_weights_file = os.path.join(external_dir, tr_weights_file)

    model_name = "VGG_semisupervised"
    experiment = "Experiment_4"
    start_fold = 0

    if "VGG" in model_name:
        normalisation_style = "VGG"
    else:
        normalisation_style = "standard"
    archi_file = "%s_archi.json" % model_name
    if model_name == "VGG_semisupervised":
        archi_file = "VGG_archi.json"

    # Set default params
    d_params = {"data_file": data_file,
                "semi_super_file": semi_super_file,
                "tr_weights_file": tr_weights_file,
                "pretr_weights_file": pretr_weights_file,
                "normalisation_style": normalisation_style,
                "objective": "categorical_crossentropy",
                "weak_labels": True,
                "model_archi_file": None,
                "img_dim": (3, 224, 224),
                "nb_classes": 10,
                "batch_size": 16,
                "n_batch_per_epoch": 100,
                "nb_epoch": 20,
                "prob": 0.8,
                "do_plot": False,
                "freeze_until": 16,
                "max_layer": 31,
                "experiment": experiment,
                "start_fold": start_fold
                }

    # Launch training
    launch_training(model_name, **d_params)

    data_file = "fullframe_VGG_test.h5"
    data_file = os.path.join(data_dir, data_file)

    tr_weights_path = "../../models/%s/%s/" % (model_name, experiment)
    model_archi_file = os.path.join(data_dir, archi_file)

    # Set default params
    d_params = {"data_file": data_file,
                "tr_weights_path": tr_weights_path,
                "model_archi_file": model_archi_file,
                "normalisation_style": normalisation_style,
                "start_fold": start_fold
                }

    # Launch submission
    launch_submission(model_name, **d_params)
