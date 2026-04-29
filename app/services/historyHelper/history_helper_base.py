from abc import ABC, abstractmethod

class HistoryHelperBase(ABC):

    # 创建历史记录，返回conversation_id
    @abstractmethod
    def create_history(self, username:str, conversation_id:str) -> None:
        pass

    # 添加用户消息到历史记录
    @abstractmethod
    def append_message_user(self, username:str, conversation_id:str, message:str):
        pass

    # 添加助手消息到历史记录
    @abstractmethod
    def append_message_assistant(self, username:str, conversation_id:str, message:str):
        pass

    # 获取单个历史记录，返回history对象
    @abstractmethod
    def get_history(self, username:str, conversation_id:str) -> dict:
        pass

    # 删除历史记录，成功删除返回 True，文件不存在返回 False
    @abstractmethod
    def delete_history(self, username:str, conversation_id:str) -> bool:
        pass
    
    # 获取所有历史记录标题列表
    @abstractmethod
    def get_all_history(self, username:str) -> list:
        pass