use qtrust::{QApplication, QPushButton, QMessageBox};

fn main() {
    let app = QApplication::new();
    
    let window = QPushButton::new("Click Me");
    window.set_window_title("Qt6 Rust App");
    
    window.clicked().connect(|| {
        let msg = QMessageBox::new();
        msg.set_text("Hello World!");
        msg.exec();
    });
    
    window.show();
    app.exec();
}
