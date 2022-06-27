# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ignite.engine import Events
import logging
from chinai.data import CSVSaver


class ClassificationSaver:
    """
    Event handler triggered on completing every iteration to save the classification predictions as CSV file.
    """

    def __init__(self, output_dir='./', filename='predictions.csv', overwrite=True,
                 batch_transform=lambda x: x, output_transform=lambda x: x, name=None):
        """
        Args:
            output_dir (str): output CSV file directory.
            filename (str): name of the saved CSV file name.
            overwrite (bool): whether to overwriting existing CSV file content. If we are not overwriting,
                then we check if the results have been previously saved, and load them to the prediction_dict.
            batch_transform (Callable): a callable that is used to transform the
                ignite.engine.batch into expected format to extract the meta_data dictionary.
            output_transform (Callable): a callable that is used to transform the
                ignite.engine.output into the form expected model prediction data.
                The first dimension of this transform's output will be treated as the
                batch dimension. Each item in the batch will be saved individually.
            name (str): identifier of logging.logger to use, defaulting to `engine.logger`.

        """
        self.saver = CSVSaver(output_dir, filename, overwrite)
        self.batch_transform = batch_transform
        self.output_transform = output_transform

        self.logger = None if name is None else logging.getLogger(name)

    def attach(self, engine):
        if self.logger is None:
            self.logger = engine.logger
        if not engine.has_event_handler(self, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        if not engine.has_event_handler(self.saver.finalize, Events.COMPLETED):
            engine.add_event_handler(Events.COMPLETED, lambda engine: self.saver.finalize())

    def __call__(self, engine):
        """
        This method assumes self.batch_transform will extract metadata from the input batch.

        """
        meta_data = self.batch_transform(engine.state.batch)
        engine_output = self.output_transform(engine.state.output)
        self.saver.save_batch(engine_output, meta_data)
