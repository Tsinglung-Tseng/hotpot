class ValidatorFactory:
    def __init__(self, session, x, y):
        self.session = session
        self.x = x
        self.y = y

    class _Validator:
        def __init__(self):
            pass

    def __call__(self):
        return self._Validator