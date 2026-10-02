-- VisionGuard AI - MySQL schema
-- Run this once against an empty database, e.g.:
--   mysql -u root -p visionguard < sql/schema.sql
-- (SQLAlchemy's init_db() also creates these automatically on first backend
--  startup, so running this file by hand is optional but useful for review.)

CREATE DATABASE IF NOT EXISTS visionguard CHARACTER SET utf8mb4;
USE visionguard;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(120) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS videos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    filepath VARCHAR(500) NOT NULL,
    processed_path VARCHAR(500),
    status VARCHAR(20) DEFAULT 'uploaded',
    duration_seconds FLOAT,
    fps FLOAT,
    total_frames INT,
    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    video_id INT NOT NULL,
    model_used VARCHAR(50),
    predicted_label VARCHAR(100),
    confidence FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS detected_objects (
    id INT AUTO_INCREMENT PRIMARY KEY,
    video_id INT NOT NULL,
    frame_number INT,
    timestamp FLOAT,
    object_name VARCHAR(100),
    confidence FLOAT,
    bbox_x1 FLOAT, bbox_y1 FLOAT, bbox_x2 FLOAT, bbox_y2 FLOAT,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS activities (
    id INT AUTO_INCREMENT PRIMARY KEY,
    video_id INT NOT NULL,
    activity_label VARCHAR(100),
    start_time FLOAT,
    end_time FLOAT,
    confidence FLOAT,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    video_id INT NOT NULL,
    event_type VARCHAR(100),
    description TEXT,
    timestamp FLOAT,
    severity VARCHAR(20) DEFAULT 'info',
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(50),
    accuracy FLOAT,
    precision_metric FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    auc FLOAT,
    trained_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);
