"""
Модуль с ошибкам data_providers
"""


class DataProviderError(Exception):
    """Общий класс ошибки DataProvider. Необходим при обработке ошибок в трейдере """
    pass


class TooManyGapsError(DataProviderError):
    """В данных слишком много пропусков.
    Ошибка будет выброшена при наличии большего количества записей None в выборке
    Порог задается при вызове метода получения данных
    """
    MESSAGE = "{0} gaps in {1} records"

    def __init__(self, gaps, total):
        self.message = self.MESSAGE.format(gaps, total)
        self.gaps = gaps
        self.total = total
        super().__init__(self.message)

    def __str__(self):
        return self.message


class UpToDateError(DataProviderError):
    """Ошибка отсутствия актуальной записи в данных
    Ошибка будет выброшена если в выборке будет отсутствовать текущие данные с биржи.
    """
    # todo Реализовать информацию об оставании
    pass
