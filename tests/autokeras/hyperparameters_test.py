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

from autokeras import hyperparameters


def test_fixed_deserialize():
    hp = hyperparameters.Fixed(10)

    deserialized = hyperparameters.deserialize(hyperparameters.serialize(hp))

    assert deserialized.value == 10


def test_choice_deserialize():
    hp = hyperparameters.Choice([10, 20], default=20)

    deserialized = hyperparameters.deserialize(hyperparameters.serialize(hp))

    assert deserialized.default == 20


def test_boolean_deserialize():
    hp = hyperparameters.Boolean(default=False)

    deserialized = hyperparameters.deserialize(hyperparameters.serialize(hp))

    assert not deserialized.default
