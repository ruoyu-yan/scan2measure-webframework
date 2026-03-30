using UnityEngine;

[RequireComponent(typeof(CharacterController))]
public class FirstPersonController : MonoBehaviour
{
    [Header("Movement")]
    [SerializeField] private float moveSpeed = 3.0f;
    [SerializeField] private float flySpeed = 5.0f;
    [SerializeField] private float sprintMultiplier = 2.0f;
    [SerializeField] private float gravity = -15.0f;

    [Header("Mouse Look")]
    [SerializeField] private float mouseSensitivity = 2.0f;
    [SerializeField] private float maxLookAngle = 85.0f;

    [Header("Ground Snap")]
    [SerializeField] private float targetEyeHeight = 1.6f;
    [SerializeField] private float snapSpeed = 8.0f;
    [SerializeField] private float groundCheckDistance = 3.0f;

    private CharacterController cc;
    private Transform cameraTransform;
    private float verticalVelocity;
    private float cameraPitch;
    private bool cursorLocked = true;
    private bool flyMode = false;

    public bool FlyMode => flyMode;
    public System.Action<bool> OnFlyModeChanged;

    void Start()
    {
        cc = GetComponent<CharacterController>();
        cameraTransform = GetComponentInChildren<Camera>().transform;
        LockCursor(true);
    }

    void Update()
    {
        HandleCursorToggle();

        if (cursorLocked)
        {
            HandleMouseLook();
            HandleModeToggle();
            if (flyMode)
                HandleFlyMovement();
            else
                HandleMovement();
        }
    }

    private void HandleCursorToggle()
    {
        // Alt key toggles cursor lock (for UI interaction)
        if (Input.GetKeyDown(KeyCode.LeftAlt))
        {
            LockCursor(!cursorLocked);
        }
    }

    private void LockCursor(bool locked)
    {
        cursorLocked = locked;
        Cursor.lockState = locked ? CursorLockMode.Locked : CursorLockMode.None;
        Cursor.visible = !locked;
    }

    private void HandleMouseLook()
    {
        float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;
        float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity;

        // Horizontal rotation on the Player object
        transform.Rotate(Vector3.up, mouseX);

        // Vertical rotation on the Camera (clamped)
        cameraPitch -= mouseY;
        cameraPitch = Mathf.Clamp(cameraPitch, -maxLookAngle, maxLookAngle);
        cameraTransform.localEulerAngles = new Vector3(cameraPitch, 0f, 0f);
    }

    private void HandleMovement()
    {
        float h = Input.GetAxisRaw("Horizontal");
        float v = Input.GetAxisRaw("Vertical");

        Vector3 moveDir = (transform.forward * v + transform.right * h).normalized;

        float speed = moveSpeed;
        if (Input.GetKey(KeyCode.LeftShift))
            speed *= sprintMultiplier;

        // Gravity
        if (cc.isGrounded)
        {
            verticalVelocity = -2.0f; // small downward force to keep grounded
        }
        else
        {
            verticalVelocity += gravity * Time.deltaTime;
        }

        Vector3 velocity = moveDir * speed;
        velocity.y = verticalVelocity;

        cc.Move(velocity * Time.deltaTime);
    }

    private void HandleModeToggle()
    {
        if (Input.GetKeyDown(KeyCode.F))
        {
            flyMode = !flyMode;
            verticalVelocity = 0f;
            OnFlyModeChanged?.Invoke(flyMode);
        }
    }

    private void HandleFlyMovement()
    {
        float h = Input.GetAxisRaw("Horizontal");
        float v = Input.GetAxisRaw("Vertical");

        Vector3 moveDir = (cameraTransform.forward * v + cameraTransform.right * h).normalized;

        float verticalInput = 0f;
        if (Input.GetKey(KeyCode.E)) verticalInput += 1f;
        if (Input.GetKey(KeyCode.Q)) verticalInput -= 1f;
        moveDir += Vector3.up * verticalInput;

        if (moveDir.sqrMagnitude > 1f)
            moveDir = moveDir.normalized;

        float speed = flySpeed;
        if (Input.GetKey(KeyCode.LeftShift))
            speed *= sprintMultiplier;

        transform.position += moveDir * speed * Time.deltaTime;
    }

    /// <summary>
    /// Temporarily unlock cursor for UI interaction (called by ToolbarUI).
    /// </summary>
    public void SetCursorLocked(bool locked)
    {
        LockCursor(locked);
    }

    public bool IsCursorLocked => cursorLocked;
}
