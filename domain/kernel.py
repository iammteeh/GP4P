from GPy.kern import Add

class Add(Add):
    def __init__(self, subkerns, name='sum'):
        _newkerns = [part for kern in subkerns for part in (kern.parts if isinstance(kern, Add) else [kern])]

        super(Add, self).__init__(_newkerns, name)
        self._exact_psicomp = self._check_exact_psicomp()