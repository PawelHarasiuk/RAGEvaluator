from google import genai

client = genai.Client(
    api_key='AIzaSyCqLLHtLpz-1w1P7bg6m9es-JeenjJ1g44'
)

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents='How does RLHF work?'
)

print(response)