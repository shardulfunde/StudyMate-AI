from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ChatBot.chatbot import cognimate_answer
from fastapi.middleware.cors import CORSMiddleware
from ChatBot.Data_templates import Last5ChatsRequest

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=False,  
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/cognimate/answer")
def get_cognimate_answer(Last5ChatsRequest: Last5ChatsRequest):
    return StreamingResponse(
        cognimate_answer(Last5ChatsRequest),
        media_type="text/plain"
    )
