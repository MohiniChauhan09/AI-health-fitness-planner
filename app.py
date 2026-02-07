import os
import streamlit as st
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq

# ------------------ ENV SETUP ------------------
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    st.error("❌ GROQ_API_KEY not found. Check your environment variable.")
    st.stop()

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Health & Fitness Planner",
    page_icon="🏋️",
    layout="wide"
)

# ------------------ MODEL ------------------
groq_model = Groq(
    id="llama-3.1-8b-instant",
    api_key=API_KEY
)


# ------------------ AGENTS ------------------
diet_agent = Agent(
    model=groq_model,
    instructions=[
        "Create a personalized daily diet plan.",
        "Include breakfast, lunch, dinner, and snacks.",
        "Mention calories and macros briefly.",
        "Keep it practical and easy to follow."
    ],
    markdown=True
)

fitness_agent = Agent(
    model=groq_model,
    instructions=[
        "Create a personalized workout plan.",
        "Include warm-up, main workout, and cooldown.",
        "Adapt to the user's fitness goal and activity level."
    ],
    markdown=True
)

planner_agent = Agent(
    model=groq_model,
    instructions=[
        "Combine diet and workout plans into one holistic plan.",
        "Add motivation, recovery tips, and lifestyle advice.",
        "Keep the tone encouraging and simple."
    ],
    markdown=True
)

# ------------------ FUNCTIONS ------------------
def generate_full_plan(name, age, weight, height, activity, diet_pref, goal):
    diet = diet_agent.run(
        f"""
        User details:
        Age: {age}
        Weight: {weight} kg
        Height: {height} cm
        Activity level: {activity}
        Diet preference: {diet_pref}
        Fitness goal: {goal}
        """
    ).content

    workout = fitness_agent.run(
        f"""
        User details:
        Age: {age}
        Weight: {weight} kg
        Height: {height} cm
        Activity level: {activity}
        Fitness goal: {goal}
        """
    ).content

    final_plan = planner_agent.run(
        f"""
        User Name: {name}

        ----------------
        DIET PLAN:
        {diet}

        ----------------
        WORKOUT PLAN:
        {workout}

        Combine these into one clear, structured health and fitness plan.
        """
    ).content

    return final_plan

# ------------------ UI ------------------
st.title("🏋️ AI Health & Fitness Planner")
st.caption("Personalized diet & workout plans powered by Groq (LLaMA-3)")

st.sidebar.header("🧾 Your Details")

name = st.sidebar.text_input("Name", "John Doe")
age = st.sidebar.number_input("Age", 10, 100, 25)
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
height = st.sidebar.number_input("Height (cm)", 120, 230, 170)

activity = st.sidebar.selectbox(
    "Activity Level",
    ["Low", "Moderate", "High"]
)

diet_pref = st.sidebar.selectbox(
    "Diet Preference",
    ["Balanced", "Vegetarian", "Keto", "Low Carb"]
)

goal = st.sidebar.selectbox(
    "Fitness Goal",
    ["Weight Loss", "Muscle Gain", "Endurance", "Flexibility"]
)

# ------------------ ACTION ------------------
if st.sidebar.button("🚀 Generate Plan"):
    with st.spinner("Creating your personalized plan..."):
        try:
            plan = generate_full_plan(
                name, age, weight, height, activity, diet_pref, goal
            )
            st.success("✅ Your plan is ready!")
            st.markdown(plan)
        except Exception as e:
            st.error("❌ Something went wrong while generating the plan.")
            st.exception(e)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("💡 *Consistency beats intensity. Stay healthy!*")





