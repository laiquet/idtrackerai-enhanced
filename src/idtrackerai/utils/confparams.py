from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ConfParams:
    """Dataclass containing all CNN hyper-parameters. These can be modified by"""

    MODEL_AREA_SD_TOLERANCE: float = 4
    MINIMUM_NUMBER_OF_CROSSINGS_TO_TRAIN_CROSSING_DETECTOR: int = 10

    MAX_IMAGES_PER_CLASS_CROSSING_DETECTOR: int = 3000
    LEARNING_RATE_DCD: float = 0.001
    BATCH_SIZE_DCD: int = 50
    BATCH_SIZE_PREDICTIONS: int = 500
    LEARNING_RATIO_DIFFERENCE_DCD: float = 0.001
    OVERFITTING_COUNTER_THRESHOLD_DCD: int = 5
    MAXIMUM_NUMBER_OF_EPOCHS_DCD: int = 30
    # Tracker with identities
    MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION: int = 3
    LEARNING_RATE_IDCNN_ACCUMULATION: float = 0.005
    VALIDATION_PROPORTION: float = 0.1
    BATCH_SIZE_IDCNN: int = 50

    LEARNING_RATIO_DIFFERENCE_IDCNN: float = 0.001

    OVERFITTING_COUNTER_THRESHOLD_IDCNN: int = 5
    OVERFITTING_COUNTER_THRESHOLD_IDCNN_FIRST_ACCUM: int = 10

    MAXIMUM_NUMBER_OF_EPOCHS_IDCNN: int = 10000

    THRESHOLD_EARLY_STOP_ACCUMULATION: float = 0.999
    THRESHOLD_ACCEPTABLE_ACCUMULATION: float = 0.9
    MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS: int = 3

    MAXIMAL_IMAGES_PER_ANIMAL: int = 3000

    RATIO_NEW: float = 0.4
    CERTAINTY_THRESHOLD: float = 0.1
    MAX_RATIO_OF_PRETRAINED_IMAGES: float = 0.9

    MIN_RATIO_OF_IMGS_ACCUMULATED_GLOBALLY_TO_START_PARTIAL_ACCUMULATION: float = 0.5
    FIXED_IDENTITY_THRESHOLD: float = 0.9
    VEL_PERCENTILE: float = 99

    def set_parameters(self, **parameters):
        """Sets parameters to self only if they are present in the class annotations.
        The set of non recognized parameters names is returned"""
        non_recognized_parameters: set[str] = set()
        for param, value in parameters.items():
            upper_param = param.upper()
            if upper_param in self.__class__.__annotations__:
                setattr(self, upper_param, value)
            else:
                non_recognized_parameters.add(param)
        return non_recognized_parameters

    def as_dict(self):
        return asdict(self)


conf = ConfParams()
