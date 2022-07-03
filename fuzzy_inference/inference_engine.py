

from fuzzy_inference.fuzzy_set import FuzzySet
from fuzzy_inference.linguistic_variable import LinguisticVariable
from fuzzy_inference.rule import Rule


class InferenceEngine:

    def __init__(self) -> None:
        self._inputvars: dict[str, LinguisticVariable] = {}
        self._outputvars: dict[str, LinguisticVariable] = {}
        self._rulebase: list[Rule] = []

    @property
    def inputvars(self):
        return self._inputvars

    @inputvars.setter
    def inputvars(self, inputvars: list[LinguisticVariable]) -> None:
        self._inputvars = {l.name: l for l in inputvars}

    @property
    def outputvars(self) -> dict[str, LinguisticVariable]:
        return self._outputvars

    @outputvars.setter
    def outputvars(self, outputvars: list[LinguisticVariable]) -> None:
        self._outputvars = {l.name: l for l in outputvars}

    def infer(self, measurements: dict):
        for k, v in measurements.items():
            self.inputvars[k].value = v

        for rule in self._rulebase:
            if rule.consequent is None:
                raise ValueError('consequent cannot be none at this stage.')
            ceil: float = min([self.inputvars[ant[0]].fuzzified[ant[1]]
                               for ant in rule.antecedents])
            outvar: LinguisticVariable = self.outputvars[rule.consequent[0]]
            # ok, I need to set value first to access fuzzified
            B = outvar.fuzzified[rule.consequent[1]]
            B_prime = FuzzySet.intersection(
                B, FuzzySet.uniform(ceil), outvar.min, outvar.max)
            outvar.output_measure = FuzzySet.union(
                outvar.output_measure, B_prime, outvar.min, outvar.max)

            # outvar.output_measure =
            # TODO: next
