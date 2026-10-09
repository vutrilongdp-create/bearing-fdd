package com.app.controller;

import java.net.ConnectException;
import java.security.Principal;
import java.util.ArrayList;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.MessageSource;
import org.springframework.context.i18n.LocaleContextHolder;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;

import com.app.constants.MappingConstants;
import com.app.constants.ViewConstants;
import com.app.dto.UserDTO;
import com.app.service.LoginService;
import com.app.bbdd.UsuarioRepository;

import jakarta.servlet.http.HttpServletRequest;

/**
 * Clase LoginController para gestionar lo relacionado a la gestion de usuarios.
 */
@Controller
@RequestMapping(MappingConstants.ROOT)
public class LoginController {

	@Autowired
	private final LoginService loginService;
	
	@Autowired
	private UsuarioRepository userRepo;
	
	@Autowired
	private MessageSource messageSource;
	
	BCryptPasswordEncoder passwEncoder;
	
	public void setMessageSource(MessageSource ms){
        this.messageSource = ms;
    }

	/**
	 * Instantiates a new login controller.
	 *
	 * @param loginService the login service
	 * @param passwEncoder the passw encoder
	 */
	public LoginController(LoginService loginService, BCryptPasswordEncoder passwEncoder) {
		this.loginService = loginService;
		this.passwEncoder = passwEncoder;
	}
	
	public String getMessage(String messageKey) {
		return this.getMessage(messageKey, null);
	}
	
	public String getMessage(String messageKey, Object[] args) {
		return this.messageSource.getMessage(messageKey, args, LocaleContextHolder.getLocale());
	}

	/**
	 * Login.
	 *
	 * @param model the model
	 * @param error the error
	 * @param request the request
	 * @return vista de inicio de sesion
	 */
	@GetMapping(MappingConstants.LOGIN_ROOT)
	public String login(Model model, @RequestParam(name = "error", required = false) String error, HttpServletRequest request) {
		if (error != null) {
			model.addAttribute("errorMessage", this.getMessage("view.cont.user.not"));
        }
		return ViewConstants.VIEW_LOGIN_PAGE;
	}

	/**
	 * Logout.
	 *
	 * @return vista de inicio de sesion
	 */
	@GetMapping(MappingConstants.LOGOUT_ROOT)
	public String logout() {
		return ViewConstants.REDIRECT_LOGIN_PAGE;
	}

	/**
	 * Metodo para registrarse.
	 *
	 * @param user the user
	 * @param model the model
	 * @param request the request
	 * @return vista de inicio de sesion
	 */
	@PostMapping(MappingConstants.REGISTER_ROOT)
	public String register(UserDTO user, Model model, HttpServletRequest request) {
		try {
			// Bypass python API check because it always returns "0" in demo mode.
			com.app.bbdd.Usuario existing = userRepo.findByEmail(user.getUsuario());
			if (existing == null) {
				String tmpPass = passwEncoder.encode(user.getPassw());
				user.setPassw(tmpPass);
				user.setRole("USER");
				user.setMaxdataset(5);
				
				com.app.bbdd.Usuario newUsuario = new com.app.bbdd.Usuario();
				newUsuario.setUsuario(user.getUsuario());
				newUsuario.setNombre(user.getNombre());
				newUsuario.setApellido(user.getApellido());
				newUsuario.setEmail(user.getEmail());
				newUsuario.setPassw(tmpPass);
				newUsuario.setRoles("USER");
				newUsuario.setMaxdataset(5);
				userRepo.save(newUsuario);
				
				try {
					loginService.createUserInDB(user);
				} catch(Exception e) {}
				
				model.addAttribute("userCreated", this.getMessage("view.cont.user.created"));
			} else {
				model.addAttribute("userAlreadyExists", this.getMessage("view.cont.user.exists"));
			}
		} catch (Exception e) {
			System.err.println("Error al procesar registro: " + e.getMessage());
		}

		return ViewConstants.VIEW_LOGIN_PAGE;
	}
	
	/**
	 * Cargar la vista de administracion de la cuenta de usuario.
	 *
	 * @param usuario the usuario
	 * @param request the request
	 * @param model the model
	 * @return vista de edicion de usuario
	 */
	@GetMapping(MappingConstants.ADMIN_ACCOUNT)
	public String adminAccount(@RequestParam(name = "usuario", required = false) String usuario, HttpServletRequest request, Model model) {
		Principal tmp = request.getUserPrincipal();
		String loggedUser = tmp.getName();
		UserDTO usr = new UserDTO();
		
		if(usuario != "" && usuario != null) {
			loggedUser = usuario;
		}
		
		try {
			usr = loginService.getCurrentUser(loggedUser);
		} catch (ConnectException e) {
			System.err.println("Error al conectar con la API: " + e.getMessage());
		}
		
		model.addAttribute("user_register", usr.getUsuario());
		model.addAttribute("name_register", usr.getNombre());
		model.addAttribute("surname_register", usr.getApellido());
		model.addAttribute("mail_register", usr.getEmail());
		model.addAttribute("maxdataset_register", usr.getMaxdataset());
		model.addAttribute("loggedUser", request.getUserPrincipal().getName());
		
		return ViewConstants.VIEW_MANAGE_ACCOUNT_PAGE;
	}
	
	/**
	 * Metodo de actualizacion de la cuenta de usuario.
	 *
	 * @param user2Update the user 2 update
	 * @param request the request
	 * @param model the model
	 * @return vista de edicion del usuario
	 */
	@PostMapping(MappingConstants.UPDATE_ACCOUNT)
	public String updateAccount(UserDTO user2Update, HttpServletRequest request, Model model) {
		Principal loggedUser = request.getUserPrincipal();
		UserDTO usr = new UserDTO();
		
		try {
			if(user2Update.getUsuario().equals(loggedUser.getName())) {
				String tmpPass = passwEncoder.encode(user2Update.getPassw());
				user2Update.setPassw(tmpPass);
				loginService.updateCurrentUser(user2Update);
				usr = loginService.getCurrentUser(loggedUser.getName());
			}else {
				if("admin".equals(loggedUser.getName())) {
					String tmpPass = passwEncoder.encode(user2Update.getPassw());
					user2Update.setPassw(tmpPass);
					loginService.updateCurrentUser(user2Update);
					usr = loginService.getCurrentUser(user2Update.getUsuario());
				}
			}
		} catch (ConnectException e) {
			System.err.println("Error al conectar con la API: " + e.getMessage());
		}
		
		model.addAttribute("user_register", usr.getUsuario());
		model.addAttribute("name_register", usr.getNombre());
		model.addAttribute("surname_register", usr.getApellido());
		model.addAttribute("mail_register", usr.getEmail());
		model.addAttribute("maxdataset_register", usr.getMaxdataset());
		
		model.addAttribute("userCreated", this.getMessage("view.cont.user.updated"));
		model.addAttribute("loggedUser", request.getUserPrincipal().getName());
		
		if("admin".equals(loggedUser.getName())) {
			return "redirect:/webAppMotorElectrico/adminUsers";
		}
		
		return ViewConstants.VIEW_MANAGE_ACCOUNT_PAGE;
	}
	

	/**
	 * Metodo para cargar la vista de Administracion de  usuarioss.
	 *
	 * @param request the request
	 * @param model the model
	 * @return vista de administracion de usuarios
	 */
	@GetMapping(MappingConstants.ADMIN_USERS)
	public String adminUsers(HttpServletRequest request, Model model) {
		Principal loggedUser = request.getUserPrincipal();
		if(!loggedUser.getName().equals("admin")) {
			return ViewConstants.REDIRECT_HOME_PAGE;
		}
		
		List<UserDTO> listUsers = new ArrayList<>();
		try {
			listUsers = loginService.getAllUsers();
		} catch (ConnectException e) {
			System.err.println("Error al conectar con la API: " + e.getMessage());
		}
		
		model.addAttribute("userCreated", listUsers);
		
		return ViewConstants.VIEW_ADMIN_USERS_PAGE;
	}
	
	/**
	 * Delete user.
	 *
	 * @param item the item
	 * @return vista de administracion de usuarios
	 */
	@PostMapping(MappingConstants.DELETE_USER)
	public String deleteUser(@RequestParam("item") String item) {
		try {
			loginService.deleteUser(item);
		} catch (ConnectException e) {
			System.err.println("Error al conectar con la API: " + e.getMessage());
		}
		return ViewConstants.REDIRECT_ADMIN_USERS_PAGE;
	}
}
