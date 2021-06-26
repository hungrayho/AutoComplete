from abc import ABC, abstractmethod


class meta_inference_engine(ABC):
    
    @abstractmethod
    def __init__():

        @abstractmethod
        def load_lang_index():
            pass
        
        @abstractmethod
        def load_model():
            pass

        pass

    @abstractmethod
    def sentence_to_vector():
        pass

    @abstractmethod
    def translate():
        pass

    @abstractmethod
    def predict():
        pass


class meta_api(ABC):
    
    @abstractmethod
    def autoc():
        pass


class meta_app(ABC):
    
    @abstractmethod
    def app():
        pass

    @abstractmethod
    def send():
        pass

    @abstractmethod
    def listen():
        pass
