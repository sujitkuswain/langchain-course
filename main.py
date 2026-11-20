from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

def main():
    information = """Elon Reeve Musk (born June 28, 1971 , Pretoria , South Africa ) is a businessman and entrepreneur known for his leadership of Tesla , SpaceX , X (formerly Twitter) , and xAI . Musk is the richest person in the world as of 2021 , with Forbes magazine estimating his fortune at $ 852 billion in February 2026. [ 1 ]

Musk was born into a wealthy family in Pretoria , South Africa, and immigrated to Canada in 1989. He holds Canadian citizenship because his mother was born there. He received a bachelor's degree from the University of Pennsylvania in 1997 and subsequently moved to California , United States , to pursue entrepreneurship. In 1995, he co-founded the software company Zip2 . After its sale in 1999 , he co-founded X.com , an online payments company that later became PayPal and was acquired by eBay in 2002. That same year, Musk also became a U.S. citizen.

In 2002, Musk founded SpaceX, a space technology company, and became its CEO and chief engineer. The company has since become a leader in innovation in reusable rockets and commercial spaceflight . As an early investor, Musk joined Tesla in 2004 and became its CEO and product architect in 2008. Tesla has become the market leader in electric vehicles . In 2015 , he co-founded OpenAI to advance artificial intelligence (AI) research, but later left. Growing dissatisfaction with the organization's direction and leadership during the AI ​​boom of the 2020s led him to found xAI . In 2022, he bought the social network Twitter , which he implemented significant changes to, and renamed it X in 2023. His other ventures include the neurotechnology company Neuralink , which he co-founded in 2016 , and The Boring Company , which he founded in 2017 and aims to develop tunnel transportation . In November 2025, Tesla approved a $1 trillion pay package for Musk, to be paid over 10 years if he meets specific goals.

Musk was the largest donor in the 2024 United States presidential election , in which he endorsed Donald Trump . After Trump took office in early 2025, Musk served as the president's senior advisor and de facto head of the Department of Government Efficiency (DOGE). After a public spat with Trump, Musk left the Trump administration and returned to running his companies. Musk is a supporter of global far-right figures, movements, and political parties. His political activities, views, and statements have made him a polarizing figure. Musk has been criticized for spreading misinformation about COVID-19 , promoting conspiracy theories , and endorsing antisemitic , racist , and transphobic comments. His acquisition of Twitter was controversial due to the subsequent increase in hate speech and the spread of misinformation on the service after he promised to reduce censorship . His role in the second Trump administration has sparked public outrage, particularly in response to DOGE. Emails he sent to sex offender Jeffrey Epstein are included in the Epstein Files , which were published in 2025–2026 and became a topic of global debate. 
"""

    summary_template = """ given the information about the {information}, tell me:
    1. who is the person
    2. what are the person's main achievements in 1 line
    """
    summary_prompt_template = PromptTemplate(
            input_variables=["information"],
            template=summary_template
        )
    
    llm=ChatOpenAI(
        model="gpt-6-luna"
    )
    
    chain = summary_prompt_template | llm
    
    response = chain.invoke({"information": information})
    
    print(response.content)


if __name__ == "__main__":
    main()
