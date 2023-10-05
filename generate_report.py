def generate_report(detected_objects):
    report = "Detected Objects:\n\n"
    for obj in detected_objects:
        class_name = obj[1]
        confidence = obj[2] * 100
        report += f"- {class_name}: {confidence:.2f}% confidence\n"

    return report

report = generate_report(detected_objects)
print(report)
