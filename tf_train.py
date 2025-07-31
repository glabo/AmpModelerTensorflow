import argparse
import traceback
from prepare import prepare
from tf_model import PedalNetTF, error_to_signal
#from tfx.components import Trainer

from tensorflow import train
from tensorflow import config as tf_config

def main(args):
    #tf_config.experimental_run_functions_eagerly(True)
    prepare(args)
    model = PedalNetTF(args)

    # grab data from pickle
    model.prepare_data()
    x_train = model.x_train
    y_train = model.y_train

    # compile the model
    model.compile(optimizer=model.optimizer(), loss=error_to_signal, metrics=['accuracy'])

    train.Checkpoint(model=model.wavenet)
    try:
        # How can we check that the model is flowing as we expect?
        model.fit(x_train, y_train, epochs=args.max_epochs, batch_size=32)
    except Exception as e:
        # Errors can take the form of printing the entire dataset, so we print them to a file
        # so that we can actually see the traceback
        with open('./failure_output/exception', mode='w') as file:
            # Get the traceback information as a string
            traceback_str = traceback.format_exc()
            # Write the exception type, message, and traceback to the file
            file.write(f"Exception Type: {type(e).__name__}\n")
            file.write(f"Exception Message: {e}\n")
            file.write(f"Traceback:\n{traceback_str}\n")
            file.write("-" * 20 + "\n") 
        print("we hit a error :( ", type(e))

    # then save model
    model.save("./model/pedalnet_tf.keras")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", nargs="?", default="data/in.wav")
    parser.add_argument("out_file", nargs="?", default="data/out.wav")
    parser.add_argument("--sample_time", type=float, default=100e-3)
    parser.add_argument("--normalize", type=bool, default=True)

    parser.add_argument("--num_channels", type=int, default=4)
    parser.add_argument("--dilation_depth", type=int, default=9)
    parser.add_argument("--num_repeat", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-3)

    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--tpu_cores", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--model", type=str, default="models/pedalnet_tf/pedalnet_model")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)

