# Hàm để gán nhãn cho thuốc
def label_medicines(text):
    # Cách tạm thời
    # Danh sách các tên thuốc
    medicine_names = [
        "Gliclazid",
        "metformin",
        "Glumeferm",
        "Losartan",
        "Savi",
        "Atorvastatin",
        "Clopidogre1",
        "RIDLOR",
        "Amlo đipin",
        "Kavasdin",
        "Lufocin",
        "Bromhexin",
        "Paracetamol",
        "Ciprofloxacin",
        "Acenocoumarol",
        "Spironolacton",
        "Metformin",
        "Giumeform",
        "Nebivoloi Khouma",
        "Rosuvastatin",
        "ROTINVAST",
        "Acarbose",
        "BLUECOSE",
    ]

    # Thay vì chỉ định thì nên gọi danh sách tên thuốc từ database và tiến hành kiểm tra

    # Kiểm tra xem `text` có nằm trong danh sách tên thuốc hay không
    if any(name in text for name in medicine_names):
        return "Medicine_Name"
    else:
        return "NaN"  # Trả về "NaN" nếu không phải tên thuốc của hệ thống
