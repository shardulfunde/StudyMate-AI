from pydantic import BaseModel,Field
class Last5ChatsRequest(BaseModel):
    last_n_chats : list[str] = Field(...,description="List of last n chats of the user including the current question.")
    current_question : str = Field(...,description="The current question asked by the user.")