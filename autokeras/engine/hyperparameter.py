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


class HyperParameter(kerastuner.engine.hyperparameters.HyperParameter):
    """Hyperparameter base class in AutoKeras.

    Specify the hyperparameter search space in the arguments of the
    initializer of the blocks. It override the Keras Tuner HyperParameter
    class to allow user not specifying the name of the HyperParameter.
    """

    def __init__(self, name="unknown", **kwargs):
        super().__init__(name=name, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.pop("conditions")
        config.pop("name")
        return config

    def add_to_hp(self, hp, name):
        """Add the HyperParameter (self) to the HyperParameters.

        # Arguments
            hp: kerastuner.HyperParameters.
            name: String. The name of the HyperParameter.
        """
        kwargs = self.get_config()
        kwargs["name"] = name
        class_name = self.__class__.__name__
        func = getattr(hp, class_name)
        return func(**kwargs)
