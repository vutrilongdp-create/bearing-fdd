package com.app;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.web.servlet.support.SpringBootServletInitializer;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;

import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;

import com.app.bbdd.Usuario;
import com.app.bbdd.UsuarioRepository;

@EnableCaching
@ConfigurationProperties
@Configuration
@SpringBootApplication
@ComponentScan(basePackages = "com.app.*")
public class WebAppMotorElectricoApplication extends SpringBootServletInitializer {

	private static final Class<WebAppMotorElectricoApplication> applicationClass = WebAppMotorElectricoApplication.class;
	private static final Log LOG = LogFactory.getLog(applicationClass);

	public static void main(String[] args) {
		try {
			LOG.info("Arranca ...");
			SpringApplication.run(applicationClass, args);
			LOG.info("Arrancado ...");
		} catch (Exception e) {
			LOG.error("Error en la aplicación ...", e);
		}
	}

	@Bean
	public CommandLineRunner initData(UsuarioRepository repo, BCryptPasswordEncoder encoder) {
		return args -> {
			if (repo.findByEmail("admin") == null) {
				Usuario admin = new Usuario();
				admin.setUsuario("admin");
				admin.setNombre("Admin");
				admin.setApellido("User");
				admin.setEmail("admin@local.com");
				admin.setPassw(encoder.encode("123456"));
				admin.setRoles("ADMIN");
				admin.setMaxdataset(50);
				repo.save(admin);
			}
			if (repo.findByEmail("liang") == null) {
				Usuario liang = new Usuario();
				liang.setUsuario("liang");
				liang.setNombre("Liang");
				liang.setApellido("User");
				liang.setEmail("liang@local.com");
				liang.setPassw(encoder.encode("123456"));
				liang.setRoles("USER");
				liang.setMaxdataset(50);
				repo.save(liang);
			}
		};
	}

}
