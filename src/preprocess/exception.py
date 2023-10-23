"""
Custom Exceptions are defined here

May be worth considering moving to different directory
"""
class input_dir_empty(Exception):
    pass

class file_number_exception(Exception):
    pass

class file_type_exception(Exception):
    pass

class datagen_not_found(Exception):
    pass

class model_not_found(Exception):
    pass

class empty_labels(Exception):
    pass

class val_error(Exception):
    pass