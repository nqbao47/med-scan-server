import re


def is_hospital(text):
    keywords = ["BV", "Bệnh viện", "Sở Y tế"]
    for keyword in keywords:
        if keyword in text:
            return True
    return False


def is_address(text):
    return "Địa chỉ" in text


def is_date(text):
    date_pattern = r'\d{1,2}\s?[/-]\s?\d{1,2}\s?[/-]\s?\d{4}'
    if re.search(date_pattern, text):
        return True
    return False


def is_diagnose(text):
    keywords = ["chẩn đoán", "triệu chứng", "bệnh"]
    for keyword in keywords:
        if keyword in text:
            return True
    return False


def is_medicines(text):
    keywords = ["Thuốc", "Tên thuốc", "liều lượng"]
    for keyword in keywords:
        if keyword in text:
            return True
    return False
