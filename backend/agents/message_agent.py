from app.models import RevenueEvent, ProposedAction

TONE_TEMPLATES = {
    "hi": {
        1: [
            "Hi {name} ji, aapka ₹{amount} ka payment pending hai. Koi issue hua tha kya? Yahan pay karein: {link}",
            "{name} ji, aapka payment nahi ho paya tha — ₹{amount}. Ek baar try karein? {link}",
            "Hi {name}, looks like your ₹{amount} payment didn't go through. Try again: {link}",
        ],
        2: [
            "{name} ji, reminder — ₹{amount} due hai. Agar abhi pura nahi de sakte, partial payment bhi chalega. Link: {link}",
            "Hi {name}, your ₹{amount} payment is still pending. We can split it if that helps: {link}",
            "{name} ji, ₹{amount} 3 din se pending hai. Please pay when you can: {link}",
        ],
        3: [
            "{name} ji, ye last reminder hai. ₹{amount} ka payment bahut overdue hai. Please aaj pay karein to avoid account issues. {link}",
            "Hi {name}, this is important — ₹{amount} needs to be paid today to keep your account active. {link}",
            "{name} ji, ₹{amount} abhi tak nahi mila. Agar aaj nahi pay kiya, account suspend ho sakta hai. {link}",
        ],
        4: [
            "Escalating your case to our team. ₹{amount} unpaid for {days} days. They'll reach out directly.",
            "{name} ji, hum aapko multiple baar contact kar chuke hain. ₹{amount} ke liye hamari team aapko call karegi.",
        ],
    },
    "en": {
        1: [
            "Hi {name}, your payment of ₹{amount} is pending. Here's the link to complete it: {link}",
            "Hey {name}, looks like ₹{amount} didn't go through. Try again here: {link}",
        ],
        2: [
            "{name}, quick reminder — ₹{amount} is still due. Partial payment works too: {link}",
            "Hi {name}, your ₹{amount} payment is overdue. You can pay here: {link}",
        ],
        3: [
            "{name}, this is your final reminder. ₹{amount} must be paid today to avoid account suspension. {link}",
            "Hi {name}, ₹{amount} has been pending for a while. Please pay immediately: {link}",
        ],
        4: [
            "Your case has been escalated. Our team will contact you regarding the ₹{amount} outstanding.",
            "{name}, we've tried reaching you multiple times. Our collections team will follow up on ₹{amount}.",
        ],
    },
}


def generate_message(
    customer_name: str,
    amount_paise: int,
    tone_level: int,
    language: str = "hi",
    link: str = "https://rzp.io/pay/demo",
    days_overdue: int = 0,
) -> str:
    import random

    lang_templates = TONE_TEMPLATES.get(language, TONE_TEMPLATES["hi"])
    templates = lang_templates.get(tone_level, lang_templates[1])
    template = random.choice(templates)

    amount_str = f"{amount_paise // 100:,}"
    first_name = customer_name.split()[0]

    return template.format(
        name=first_name,
        amount=amount_str,
        link=link,
        days=days_overdue,
    )


def generate_reauth_message(customer_name: str, amount_paise: int, language: str = "hi") -> str:
    amount_str = f"{amount_paise // 100:,}"
    first_name = customer_name.split()[0]

    if language == "hi":
        return (
            f"Hi {first_name} ji, aapka ₹{amount_str} ka recurring payment ruk gaya hai. "
            f"RBI rules ke according, ₹15,000 se zyada ke liye fresh authorization zaroori hai. "
            f"Yahan se authorize karein: https://rzp.io/mandate/reauth/{first_name.lower()}"
        )
    return (
        f"Hi {first_name}, your recurring payment of ₹{amount_str} was paused. "
        f"Per RBI guidelines, amounts above ₹15,000 need fresh authorization. "
        f"Please authorize here: https://rzp.io/mandate/reauth/{first_name.lower()}"
    )


def apply_message_to_action(
    event: RevenueEvent,
    proposed: ProposedAction,
) -> ProposedAction:
    if proposed.action == "send_dunning_message":
        days = event.metadata.days_overdue or 0
        text = generate_message(
            customer_name=event.customer.name,
            amount_paise=event.amount,
            tone_level=proposed.message_tone_level,
            language=event.customer.language_pref,
            days_overdue=days,
        )
        proposed.message_text = text

    elif proposed.action == "re_authorize_mandate":
        text = generate_reauth_message(
            customer_name=event.customer.name,
            amount_paise=event.amount,
            language=event.customer.language_pref,
        )
        proposed.message_text = text

    elif proposed.action == "send_payment_link":
        text = generate_message(
            customer_name=event.customer.name,
            amount_paise=event.amount,
            tone_level=1,
            language=event.customer.language_pref,
        )
        proposed.message_text = text

    return proposed
