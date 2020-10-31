# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kerastuner
import tensorflow as tf

from autokeras.engine import hyperparameter


class Choice(
    hyperparameter.HyperParameter, kerastuner.engine.hyperparameters.Choice
):
    """Choice of one value among a predefined set of possible values.

    # Arguments
        values: List of possible values. Values must be int, float,
            str, or bool. All values must be of the same type.
        ordered: Whether the values passed should be considered to
            have an ordering. This defaults to `True` for float/int
            values. Must be `False` for any other values.
        default: Default value to return for the parameter.
            If unspecified, the default value will be:
            - None if None is one of the choices in `values`
            - The first entry in `values` otherwise.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, values, ordered=None, default=None, **kwargs):
        super().__init__(values=values, ordered=ordered, default=default, **kwargs)


class Fixed(hyperparameter.HyperParameter, kerastuner.engine.hyperparameters.Fixed):
    """Fixed, untunable value.

    # Arguments
        value: Value to use (can be any JSON-serializable
            Python type).
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, value, **kwargs):
        super().__init__(value=value, **kwargs)


class Boolean(
    hyperparameter.HyperParameter, kerastuner.engine.hyperparameters.Boolean
):
    """Choice between True and False.

    # Arguments
        default: Default value to return for the parameter.
            If unspecified, the default value will be False.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, default, **kwargs):
        super().__init__(default=default, **kwargs)


def serialize(obj):
    return tf.keras.utils.serialize_keras_object(obj)


def deserialize(config, custom_objects=None):
    return tf.keras.utils.deserialize_keras_object(
        config,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name="hypermodels",
    )
