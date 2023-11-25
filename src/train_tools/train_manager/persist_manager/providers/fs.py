import os
import pickle


class FsDataProvider:
    """Формирует рабочий путь вида path + dir + alias
     Если заданынй path не существует, то используется значение fallback

    """
    FALLBACK_PATH = "./"  # Путь на всякий сулчай, если сетевой недоступен
    DIR = "train_data_test"  # Рабочая директория, куда все будет сливаться

    def __init__(self, alias, path, work_dir, name_prefix):
        self.alias = alias
        self.path = path
        self.work_dir = work_dir
        self.name_prefix = name_prefix
        self.last_work_path = None

    def get_path(self, *args):
        """Проверяет наличие директории и при необходимости ее создает"""

        if os.path.exists(self.path):
            work_path = self.path
        else:
            work_path = self.FALLBACK_PATH

        if self.last_work_path != work_path:
            print(f"New work path: {work_path}")
        self.last_work_path = work_path

        path = os.path.join(work_path, self.DIR, self.alias, self.work_dir,  *list(map(str, args)))
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_file_path(self, name):
        return os.path.join(self.get_path(), self.name_prefix + "-" + str(name) + ".pkl")

    def put(self, name, data):
        file_path = self.get_file_path(name)
        try:
            with open(file_path, 'wb') as stream:
                pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
                # stream.write(data)
        except Exception as e:
            print(f"Ошибка при сохранении файла {file_path}")
            print(e)

    def get(self, name):
        file_path = self.get_file_path(name)
        try:
            with open(file_path, "rb") as stream:
                data = pickle.load(stream)
                # data = stream.read()
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}")
            print(e)
            data = None
        return data
