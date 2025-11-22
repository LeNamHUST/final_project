# from fastapi.testclient import TestClient
# from api import app

# client = TestClient(app)

# def test_api():
#     with open("/home/namdao/projects/final_project/data/test/test/0a2229abced7.jpg", "rb") as img:
#         response = client.post(
#             "/retinal_diesase", files={"file":("sample_image.jpg", img, "image/jpeg")}
#         )
#         assert response.status_code == 200, "loi api"
#         json_data = response.json()
#         assert "message" in json_data, "api khong tra ra ket qua"
#         assert len(json_data["message"]) > 0, "model khong du doan duoc, xem lai"