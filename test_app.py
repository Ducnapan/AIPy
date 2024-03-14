import pytest
from flask import url_for
from Facial_Recognition_App import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200

def test_video_feed(client):
    response = client.get('/video_feed')
    assert response.status_code == 200

