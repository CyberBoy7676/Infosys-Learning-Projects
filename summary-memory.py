# for run code : c:\users\adilr\anaconda3\python.exe .\buffer-memory.py
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain



# Intialize th echat model
chat = ChatGoogleGenerativeAI(
        model = 'gemini-1.5-flash',
        api_key = 'AIzaSyBsk9teXzKWVHeOKHN_PbVBAVZ3CI5oV5k'
    )

# it creates a  memory for the conversation
memory = ConversationSummaryMemory(llm=chat)

# it creates a conversation chain with the chat model and the memory
conversation = ConversationChain(
    llm = chat,
    memory = memory,
)

while True:
    # Here we are taking user input
    user_input = input("\nYou( Human ) : ")

    # Check for exit command
    if user_input.lower() in ['over','exit']:
        print("Khatam-Tata_bye_bye") 

        # print the summaries history
        print(conversation.memory.buffer)
        break

    # here we are getting the response from the AI
    response = conversation.predict(input=user_input)
    print("\nAI Generated : ", response)