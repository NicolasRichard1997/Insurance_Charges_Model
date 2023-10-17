import unittest

from insurance_charges_model.prediction.model import InsuranceChargesModel
from insurance_charges_model.prediction.schemas import InsuranceChargesModelInput, \
    InsuranceChargesModelOutput, SexEnum, RegionEnum
from pydantic import ValidationError


class ModelTests(unittest.TestCase):

    def test_model(self):
        # arrange
        model = InsuranceChargesModel()
        inpt = dict(age=35, sex="male", bmi=20.0, children=1, smoker=False, region="northeast")

        # act
        prediction = model.predict(InsuranceChargesModelInput(**inpt))

        # assert
        self.assertTrue(type(prediction) is InsuranceChargesModelOutput)

    def test_model_with_wrong_input_type(self):
        # arrange
        model = InsuranceChargesModel()
        inpt = dict(age=35.0, sex="asdas", bmi=20.0, children=1, smoker=False, region=RegionEnum.northeast)

        # act, assert
        with self.assertRaises(ValidationError):
            prediction = model.predict(InsuranceChargesModelInput(**inpt))
            

if __name__ == '__main__':
    unittest.main()
